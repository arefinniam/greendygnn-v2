import torch as th
import time
import threading
from collections import deque
from typing import Dict, Optional


class FeatureCache:
    """Double-buffered hot-feature cache with per-owner fetch instrumentation.

    Concurrency contract (V2):
      - The worker thread is the only reader of the ACTIVE buffer
        (get_features) and the only caller of swap_buffers().
      - The builder thread is the only writer, and only ever writes the
        PENDING buffer via set_write_buffer_state().
      - swap_buffers() and set_write_buffer_state() are serialized by
        self._swap_lock; a builder commit passes the active index it
        captured at build start and the commit is REFUSED (returns False)
        if a swap happened in between, so a stale build can never clobber
        the buffer the worker is reading (spec V2 I1/P1).
    """

    def __init__(self, g, n_hot, device, dist_lock=None,
                 track_detailed_metrics=False, owner_of=None):
        self.part_id = None
        self.n_hot = n_hot
        self.dist_lock = dist_lock
        self.track_detailed_metrics = track_detailed_metrics

        # Cache feature metadata once to avoid repeated DistTensor lookups
        self.feat_dim = g.ndata["features"].shape[1]
        self.feat_dtype = g.ndata["features"].dtype
        self._elem_size = th.tensor([], dtype=self.feat_dtype).element_size()

        # Double buffering: [Buffer 0, Buffer 1]
        self.cache_nodes = [
            th.tensor([], dtype=th.long, device=device),
            th.tensor([], dtype=th.long, device=device)
        ]
        self.cache_features = [
            th.empty((0, self.feat_dim), device=device),
            th.empty((0, self.feat_dim), device=device)
        ]
        # int32: positions are < n_hot << 2**31, and the map is O(num_nodes),
        # so int32 halves its memory and per-step gather bandwidth. The small
        # per-batch position slices are cast to long at the point of feature
        # indexing (torch requires long there on some versions).
        self.cache_idx = [
            th.full((g.num_nodes(),), -1, dtype=th.int32, device=device),
            th.full((g.num_nodes(),), -1, dtype=th.int32, device=device)
        ]

        self.active_idx = 0  # 0 or 1
        self._swap_lock = threading.Lock()

        self.pb = None
        try:
            self.pb = g.get_partition_book()
        except AttributeError:
            self.pb = None
        self.local_owner = None
        if self.pb is not None:
            self.local_owner = getattr(self.pb, "partid", None)
        if self.local_owner is None:
            self.local_owner = g.rank()

        # owner_of(nodes: LongTensor) -> LongTensor of partition ids.
        # Injected for testability; falls back to the partition book.
        if owner_of is not None:
            self._owner_of_fn = owner_of
        elif self.pb is not None:
            self._owner_of_fn = self.pb.nid2partid
        else:
            self._owner_of_fn = None

        self.current_window_start: Optional[int] = None

        # Basic counters
        self.total_fetches = 0
        self.cache_hits = 0
        self.remote_cache_hits = 0
        self.remote_misses = 0
        self.local_misses = 0

        # Step-level counters (avoid per-batch hasattr checks)
        self._step_remote_hits = 0
        self._step_remote_misses = 0

        # Per-owner miss tracking (kept for backward compat with optisched path)
        self._owner_miss_counts = {}  # {part_id: count} per window
        self._last_fetch_time_s = 0.0  # time spent in last DistTensor fetch
        self._total_fetch_time_s = 0.0
        self._fetch_count = 0

        # === V2 I1: per-owner fetch event log ===
        # each event: (t_wall, owner_pid, rows, bytes, rtt_s)
        # single writer (worker thread); readers snapshot under _events_lock.
        self.fetch_events = deque(maxlen=4096)
        self._events_lock = threading.Lock()
        # cumulative remote-fetch counters for the profiler (I5)
        self.remote_fetch_ops = 0
        self.remote_rows_fetched = 0
        self.remote_bytes_fetched = 0

    # ------------------------------------------------------------------ #
    # Double buffer management
    # ------------------------------------------------------------------ #
    def get_write_buffer_indices(self):
        """Return (nodes, features, idx_map) for the pending (write) buffer"""
        write_idx = 1 - self.active_idx
        return (self.cache_nodes[write_idx],
                self.cache_features[write_idx],
                self.cache_idx[write_idx])

    def set_write_buffer_state(self, nodes, features, expected_active_idx=None):
        """Commit the pending buffer contents.

        `expected_active_idx` is the active index the builder captured when
        it STARTED the build. If a swap happened since (worker moved on),
        the commit is refused and False is returned so a stale build can
        never write into the buffer the worker is actively reading.
        Returns True on a successful commit.
        """
        with self._swap_lock:
            if (expected_active_idx is not None
                    and expected_active_idx != self.active_idx):
                return False
            write_idx = 1 - self.active_idx
            idx_map = self.cache_idx[write_idx]

            # O(k) reset instead of an O(num_nodes) fill: the map invariant is
            # "non--1 exactly at cache_nodes[write_idx]", so scattering -1 at
            # the previous commit's nodes clears every stale entry.
            prev_nodes = self.cache_nodes[write_idx]
            if prev_nodes.numel() > 0:
                idx_map[prev_nodes] = -1

            self.cache_nodes[write_idx] = nodes
            self.cache_features[write_idx] = features
            if len(nodes) > 0:
                idx_map[nodes] = th.arange(
                    len(nodes), dtype=idx_map.dtype, device=idx_map.device)
            return True

    def swap_buffers(self):
        """Make the pending buffer active"""
        with self._swap_lock:
            self.active_idx = 1 - self.active_idx

    # ------------------------------------------------------------------ #
    # Owner helpers
    # ------------------------------------------------------------------ #
    def owner_of(self, nodes):
        """Map node ids -> owner partition ids (LongTensor), or None."""
        if self._owner_of_fn is None:
            return None
        try:
            owners = self._owner_of_fn(nodes)
            if not isinstance(owners, th.Tensor):
                owners = th.as_tensor(owners)
            return owners.long()
        except Exception as e:
            print(f"[FeatureCache] owner_of failed ({type(e).__name__}: {e}); "
                  f"falling back to combined fetch")
            return None

    # ------------------------------------------------------------------ #
    # Fetch path
    # ------------------------------------------------------------------ #
    def get_features(self, input_nodes, g, device, remote_mask, batch_idx):
        if self.part_id is None:
            self.part_id = g.rank()

        # input_nodes and remote_mask are already on CPU from BackgroundSampler
        input_nodes_cpu = input_nodes
        remote_mask_cpu = remote_mask

        n_inputs = input_nodes_cpu.numel()

        # Allocate output tensor on target device
        out = th.empty((n_inputs, self.feat_dim), dtype=self.feat_dtype, device=device)

        self.total_fetches += n_inputs

        # Use ACTIVE buffer - capture index locally for consistency
        idx = self.active_idx
        active_nodes = self.cache_nodes[idx]
        active_features = self.cache_features[idx]
        active_idx_map = self.cache_idx[idx]

        if active_nodes.numel() > 0:
            input_nodes_dev = input_nodes_cpu.to(device)
            pos = active_idx_map[input_nodes_dev]
            # Single device sync per step: the hit/miss split must reach the
            # CPU to select the nodes to fetch. Everything else (counters,
            # remote/local split) is CPU-side — no .any()/int() syncs and no
            # mask uploads.
            hit_mask_cpu = (pos >= 0).to("cpu")
            hit_rows_cpu = th.nonzero(hit_mask_cpu, as_tuple=True)[0]

            if hit_rows_cpu.numel() > 0:
                hit_rows_dev = hit_rows_cpu.to(device)
                # .long() only on the small per-batch slice; the O(N) map
                # stays int32.
                out[hit_rows_dev] = active_features[pos[hit_rows_dev].long()]

            n_hits = int(hit_rows_cpu.numel())
            n_remote_hits = int((hit_mask_cpu & remote_mask_cpu).sum())
            self.cache_hits += n_hits
            self.remote_cache_hits += n_remote_hits
            self._step_remote_hits += n_remote_hits

            miss_mask_cpu = ~hit_mask_cpu
        else:
            miss_mask_cpu = th.ones(n_inputs, dtype=th.bool)

        if miss_mask_cpu.any():
            remote_miss_count = int((miss_mask_cpu & remote_mask_cpu).sum())
            if remote_miss_count:
                self.remote_misses += remote_miss_count
                self._step_remote_misses += remote_miss_count

            local_miss_count = int((miss_mask_cpu & ~remote_mask_cpu).sum())
            if local_miss_count:
                self.local_misses += local_miss_count

            miss_rows_cpu = th.nonzero(miss_mask_cpu, as_tuple=True)[0]
            miss_nodes = input_nodes_cpu[miss_rows_cpu]
            # Misses are copied host->device per owner straight into `out`
            # (no CPU staging tensor, no full-mask upload).
            self._fetch_by_owner(miss_nodes, g, out=out,
                                 dest_rows_cpu=miss_rows_cpu, device=device)

        return out

    def _fetch_by_owner(self, miss_nodes, g, out=None, dest_rows_cpu=None,
                        device=None):
        """Pull missing features, split by owner partition, timing each pull.

        This is the I1 instrument: each remote owner's pull is timed
        separately and recorded as a fetch event (t_wall, owner, rows,
        bytes, rtt_s). The pulls are DELIBERATELY serial per owner — the
        per-owner rtt attribution that feeds sigma_hat/khat depends on it;
        do not parallelize without redesigning the congestion signal.

        When `out`/`dest_rows_cpu`/`device` are given, each owner's rows are
        copied host->device directly into `out[dest_rows_cpu[sel]]` (no CPU
        staging tensor). Otherwise a CPU tensor of the missed rows is
        returned (legacy path). The recorded rtt covers the pull only; the
        host->device copy is outside the rtt timer (but inside the
        last_fetch_time window). Falls back to a single combined pull when
        no owner mapping is available.
        """
        n_miss = miss_nodes.numel()
        owners = self.owner_of(miss_nodes)
        t_total0 = time.perf_counter()

        scatter = out is not None and dest_rows_cpu is not None
        feats = None if scatter else th.empty(
            (n_miss, self.feat_dim), dtype=self.feat_dtype)

        if owners is None:
            # Legacy combined pull
            if self.dist_lock:
                with self.dist_lock:
                    pulled = g.ndata["features"][miss_nodes]
            else:
                pulled = g.ndata["features"][miss_nodes]
            if scatter:
                out[dest_rows_cpu.to(device)] = pulled.to(
                    device, non_blocking=True)
            else:
                feats = pulled
            self._last_fetch_time_s = time.perf_counter() - t_total0
            self._total_fetch_time_s += self._last_fetch_time_s
            self._fetch_count += 1
            return feats

        uniq_owners = owners.unique().tolist()
        now = time.time()
        for pid in uniq_owners:
            sel = (owners == pid)
            nodes_o = miss_nodes[sel]
            rows = int(nodes_o.numel())
            if rows == 0:
                continue
            t0 = time.perf_counter()
            if self.dist_lock:
                with self.dist_lock:
                    feats_o = g.ndata["features"][nodes_o]
            else:
                feats_o = g.ndata["features"][nodes_o]
            rtt = time.perf_counter() - t0
            if scatter:
                out[dest_rows_cpu[sel].to(device)] = feats_o.to(
                    device, non_blocking=True)
            else:
                feats[sel] = feats_o

            if pid != self.local_owner:
                nbytes = rows * self.feat_dim * self._elem_size
                with self._events_lock:
                    self.fetch_events.append((now, int(pid), rows, nbytes, rtt))
                    self.remote_fetch_ops += 1
                    self.remote_rows_fetched += rows
                    self.remote_bytes_fetched += nbytes
                # per-owner miss counts kept for the legacy optisched path
                self._owner_miss_counts[int(pid)] = \
                    self._owner_miss_counts.get(int(pid), 0) + rows

        self._last_fetch_time_s = time.perf_counter() - t_total0
        self._total_fetch_time_s += self._last_fetch_time_s
        self._fetch_count += 1
        return feats

    # ------------------------------------------------------------------ #
    # V2 I1 read APIs
    # ------------------------------------------------------------------ #
    def snapshot_fetch_events(self, clear=False):
        """Copy of the fetch-event ring buffer (thread-safe)."""
        with self._events_lock:
            events = list(self.fetch_events)
            if clear:
                self.fetch_events.clear()
        return events

    def get_owner_latency_stats(self, last_n=None):
        """Per-owner aggregate of recent fetch events.

        Returns {pid: {"n": int, "rows": int, "bytes": int,
                       "mean_rtt": float, "mean_rtt_per_row": float}}
        """
        events = self.snapshot_fetch_events()
        if last_n is not None:
            events = events[-last_n:]
        stats = {}
        for (_t, pid, rows, nbytes, rtt) in events:
            s = stats.setdefault(pid, [0, 0, 0, 0.0, 0.0])
            s[0] += 1
            s[1] += rows
            s[2] += nbytes
            s[3] += rtt
        out = {}
        for pid, (n, rows, nbytes, rtt_sum, _) in stats.items():
            out[pid] = {
                "n": n, "rows": rows, "bytes": nbytes,
                "mean_rtt": rtt_sum / n if n else 0.0,
                "mean_rtt_per_row": (rtt_sum / rows) if rows else 0.0,
            }
        return out

    def get_remote_fetch_counters(self):
        """Cumulative remote fetch counters (ops, rows, bytes) — I5."""
        with self._events_lock:
            return (self.remote_fetch_ops, self.remote_rows_fetched,
                    self.remote_bytes_fetched)

    # ------------------------------------------------------------------ #
    # Legacy read APIs (kept: train_optisched.py depends on these)
    # ------------------------------------------------------------------ #
    def get_owner_miss_counts(self):
        """Per-owner miss counts since last call (resets internal state)."""
        counts = dict(self._owner_miss_counts)
        self._owner_miss_counts.clear()
        return counts

    def get_avg_fetch_time(self):
        """Return average DistTensor fetch time in seconds."""
        if self._fetch_count == 0:
            return 0.0
        return self._total_fetch_time_s / self._fetch_count

    def get_last_fetch_time(self):
        return self._last_fetch_time_s

    def get_stats(self):
        """Return basic statistics"""
        total_requests = max(1, self.total_fetches)
        total_remote_requests = max(1, self.remote_cache_hits + self.remote_misses)

        return {
            'total_fetches': self.total_fetches,
            'cache_hits': self.cache_hits,
            'remote_cache_hits': self.remote_cache_hits,
            'remote_misses': self.remote_misses,
            'local_misses': self.local_misses,
            'cache_hit_rate': self.cache_hits / total_requests,
            'remote_cache_hit_rate': self.remote_cache_hits / total_remote_requests,
            'remote_miss_rate': self.remote_misses / total_requests,
            'local_miss_rate': self.local_misses / total_requests
        }

    def get_step_remote_hit_rate(self):
        """
        Get remote cache hit rate for current step and reset step counters.
        Returns hit rate as percentage (0-100).
        """
        total = self._step_remote_hits + self._step_remote_misses
        hit_rate = (self._step_remote_hits / total * 100) if total > 0 else 0.0

        # Reset for next step
        self._step_remote_hits = 0
        self._step_remote_misses = 0

        return hit_rate
