import threading, queue, time, torch as th
from collections import deque

try:
    from optisched.live_alloc import select_owner_budgeted
except Exception:  # keep baselines (rapidgnn/greendygnn/default) importable regardless
    select_owner_budgeted = None

# V2 analytic allocator (Agent S, spec I3). Falls back to optisched.live_alloc.
try:
    from alloc import select_owner_budgets as _select_owner_budgets_v2
except Exception:
    _select_owner_budgets_v2 = None


class SharedBuffer:
    """Thread-safe buffer for passing batches between Sampler and Prefetcher"""
    def __init__(self, capacity=500):
        self.capacity = capacity
        self.buffer = deque()
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        self.finished = False

    def put(self, item):
        with self.lock:
            while len(self.buffer) >= self.capacity and not self.finished:
                self.not_full.wait()
            if self.finished:
                return
            self.buffer.append(item)
            self.not_empty.notify()

    def get(self):
        with self.lock:
            while not self.buffer and not self.finished:
                self.not_empty.wait()
            if not self.buffer and self.finished:
                return None
            item = self.buffer.popleft()
            self.not_full.notify()
            return item

    def peek_n(self, n):
        """Return the next n items without removing them.

        n is clamped to the buffer capacity so a request that could never
        be satisfied (worker far ahead of the sampler) cannot deadlock the
        caller (spec V2 P3).
        """
        with self.lock:
            n = min(n, self.capacity)
            # Wait until we have enough items or finished
            while len(self.buffer) < n and not self.finished:
                self.not_empty.wait()

            count = min(n, len(self.buffer))
            return list(self.buffer)[:count]

    def size(self):
        with self.lock:
            return len(self.buffer)

    def mark_finished(self):
        with self.lock:
            self.finished = True
            self.not_empty.notify_all()
            self.not_full.notify_all()


class BackgroundSampler(threading.Thread):
    """Producer thread that runs the DGL sampler and fills the SharedBuffer"""
    def __init__(self, sampler, buffer, g, local_mask, start_batch_id=0, dist_lock=None, initial_iter=None, num_epochs=1, digest_path=None, trace_dump_dir=None):
        super().__init__(daemon=True)
        self.sampler = sampler
        self.buffer = buffer
        self.g = g
        self.local_mask = local_mask
        self.current_batch_id = start_batch_id
        self.dist_lock = dist_lock
        self.initial_iter = initial_iter
        self.num_epochs = num_epochs
        self.running = True
        self.error = None
        # Optional rolling digest of the sampled remote-access stream, for
        # live-vs-collected trace equivalence checks (trace_digest.py).
        self.digest_path = digest_path
        self._digest = None
        if digest_path:
            from trace_digest import new_digest
            self._digest = new_digest()
        # Optional live trace capture (Layer-1 exactness): distributed neighbor
        # sampling is nondeterministic across runs (server-side RNG draw order
        # depends on request interleaving), so a separately collected trace can
        # never be exact — the trace of a run must be dumped BY that run.
        # Epoch attribution needs a clean epoch grid, so dumping is disabled
        # when initial_iter splits epoch 0.
        self.trace_dump_dir = trace_dump_dir
        if trace_dump_dir and initial_iter is not None:
            print("BackgroundSampler: trace_dump disabled (initial_iter in use)")
            self.trace_dump_dir = None
        if self.trace_dump_dir:
            import os
            os.makedirs(self.trace_dump_dir, exist_ok=True)
        self._dump_pb = None
        self._dump_epoch = 0
        self._dump_files = []
        self._cur_epoch_nodes = []   # this epoch's per-batch remote-id arrays
        self._epochs_nodes = []      # sealed per-epoch lists, written at the end

    def run(self):
        try:
            # If initial_iter is provided we finish it as the remainder of epoch 0,
            # then iterate from epoch 1. Otherwise iterate all epochs from scratch.
            if self.initial_iter:
                self._consume_iterator(self.initial_iter)
            start_epoch = 1 if self.initial_iter else 0
            for epoch in range(start_epoch, self.num_epochs):
                if not self.running: break
                iterator = iter(self.sampler)
                self._consume_iterator(iterator)
                self._seal_trace_epoch()      # cheap: just move the list

        except Exception as e:
            self.error = e
            print(f"BackgroundSampler error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.trace_dump_dir is not None:
                try:
                    if self._cur_epoch_nodes:   # partial epoch on early stop
                        self._seal_trace_epoch()
                    # All owner-lookup RPCs + npz writes happen HERE, after the
                    # timed epoch loop, so they never inflate step/energy
                    # metrics (measured 2026-07-17: doing them per-epoch inside
                    # the loop cost ~50% training wall time).
                    for ep_nodes in self._epochs_nodes:
                        self._write_trace_epoch(ep_nodes)
                    self._write_trace_meta()
                except Exception as e:
                    print(f"BackgroundSampler trace dump finalize failed: {e}")
            if self._digest is not None:
                try:
                    from trace_digest import write_digest
                    write_digest(self.digest_path, self._digest,
                                 self.current_batch_id)
                except Exception as e:
                    print(f"BackgroundSampler digest write failed: {e}")
            self.buffer.mark_finished()

    def _seal_trace_epoch(self):
        """Cheap: hand this epoch's batch arrays to the write list, reset."""
        if self.trace_dump_dir is None:
            return
        self._epochs_nodes.append(self._cur_epoch_nodes)
        self._cur_epoch_nodes = []

    def _write_trace_epoch(self, bn):
        """Heavy (concat + owner RPC + npz). Called AFTER the timed loop."""
        import os
        import numpy as np
        from optisched.trace import Trace
        if self._dump_pb is None:
            self._dump_pb = self.g.get_partition_book()
        pb = self._dump_pb
        pid, P = int(self.g.rank()), int(pb.num_partitions())
        total = int(sum(len(x) for x in bn))
        nodes = np.concatenate(bn) if total else np.empty(0, np.int64)
        if total:
            import torch as _th
            owners = pb.nid2partid(_th.from_numpy(nodes)).numpy().astype("int32")
        else:
            owners = np.empty(0, np.int32)
        tr = Trace(
            nodes=nodes, owners=owners,
            offsets=np.cumsum([0] + [len(x) for x in bn], dtype=np.int64),
            num_partitions=P, local_rank=pid)
        path = os.path.join(self.trace_dump_dir,
                            f"trace_part{pid}_ep{self._dump_epoch:03d}.npz")
        tr.save(path)
        self._dump_files.append(os.path.basename(path))
        self._dump_epoch += 1

    def _write_trace_meta(self):
        import json
        import os
        pid = int(self.g.rank())
        meta = {"pid": pid, "source": "live",
                "epochs_dumped": self._dump_epoch,
                "n_batches": int(self.current_batch_id),
                "digest": self._digest.hexdigest() if self._digest else None,
                "files": self._dump_files}
        with open(os.path.join(self.trace_dump_dir,
                               f"trace_part{pid}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def _consume_iterator(self, iterator):
        while self.running:
            try:
                if self.dist_lock:
                    with self.dist_lock:
                        input_nodes, seeds, blocks = next(iterator)
                else:
                    input_nodes, seeds, blocks = next(iterator)
            except StopIteration:
                break
            except Exception as e:
                print(f"BackgroundSampler iterator error: {e}")
                break

            # Process batch
            remote_mask = ~self.local_mask[input_nodes]
            if self._digest is not None:
                from trace_digest import update_digest
                update_digest(self._digest, self.current_batch_id,
                              input_nodes[remote_mask])
            if self.trace_dump_dir is not None:
                # Hot path: capture remote ids only (cheap CPU copy). All
                # concat/owner-lookup/npz work is deferred to after the timed
                # loop (see run()); doing any of it per-epoch inside the loop
                # cost ~50% of training wall time (measured 2026-07-17).
                self._cur_epoch_nodes.append(
                    input_nodes[remote_mask].numpy().astype("int64"))

            # Access labels safely
            if self.dist_lock:
                with self.dist_lock:
                    labels = self.g.ndata["labels"][seeds].cpu()
            else:
                labels = self.g.ndata["labels"][seeds].cpu()

            # Store as tuple
            batch_data = (input_nodes, seeds, blocks, remote_mask, labels)
            self.buffer.put(batch_data)
            self.current_batch_id += 1

    def stop(self):
        self.running = False


class BatchPrefetcher:
    """Async prefetch pipeline with a double-buffered window cache.

    V2 build/commit/swap protocol (spec P1/P2):
      - Build requests carry a monotonically increasing sequence number
        (build_seq). The builder captures (seq, start_batch) under
        self.gen_lock, builds WITHOUT the lock, then commits only if its
        seq is still the newest request; commits additionally pass the
        active-buffer index captured at build start so cache.set_write_
        buffer_state refuses the write if a swap happened meanwhile.
      - The worker swaps ONLY when next_window_ready fires AND
        ready_seq == build_seq. On timeout it does NOT swap: it keeps
        serving the stale-but-consistent active buffer, extends the window
        validity, and counts stale_window_count. A late build is consumed
        at the next boundary (late_window_count if its planned start
        differs from the actual boundary).
    """

    def __init__(self, g, device, cache, window_size, shared_buffer, total_batches, batches_per_epoch, max_batches=8, synchronous_cache=False, n_classes=None, window_hot_nodes=None, dist_lock=None, initial_data=None):
        self.g, self.device, self.cache = g, device, cache
        self.n_classes = n_classes
        self.part_id = g.rank()
        self.window_size = window_size
        self.shared_buffer = shared_buffer # Replaces presample_db. Can be SharedBuffer or list.
        self.total_batches = total_batches
        self.batches_per_epoch = batches_per_epoch
        self.synchronous_cache = synchronous_cache
        self.window_hot_nodes = window_hot_nodes or {}
        self.dist_lock = dist_lock
        self.initial_data = initial_data if isinstance(initial_data, list) else None
        self.initial_data_iter = iter(initial_data) if initial_data else None

        try:
            self.pb = g.get_partition_book()
        except AttributeError:
            self.pb = None

        if not self.synchronous_cache:
            self.buffer = queue.Queue(maxsize=max_batches)
            self.stop_event = threading.Event()

            # Parallel cache-building primitives (V2 protocol)
            self.next_window_ready = threading.Event()
            self.build_next_window = threading.Event()
            self.cache_builder_thread = None
            self.gen_lock = threading.Lock()
            self.build_seq = 0           # newest issued request
            self.ready_seq = -1          # newest committed build
            self._pending_build = None   # (seq, start_batch)
            self.last_built = None       # (seq, start_batch, win_len)
            self.builder_timeout_s = 30.0

        self.cost_weights = None  # {part_id: weight} legacy cost-weighted selection
        # OptiSched floor-guided owner-aware allocation (Thm G): {part_id: kappa_m}.
        self.owner_kappa = None
        self.last_owner_budgets = None
        self.num_owners = None
        if self.pb is not None:
            self.num_owners = getattr(self.pb, "num_partitions", lambda: None)()
        self.cache_valid_from_batch = 0
        self.cache_valid_until_batch = 0
        self.current_batch_idx = 1 # 1-based index
        self.total_popped = 0 # Track total batches consumed for offset calc
        self.count_lock = threading.Lock()  # guards total_popped (spec P2)

        # OptiSched non-uniform schedule (optional)
        self.schedule = None

        # === V2 controller integration (I2/I4) ===
        self.controller = None
        self.rl_enabled = True          # False => ignore d.W (static-W ablation)
        self.uniform_alloc = False      # True => ignore khat (uniform ablation)
        self.controller_khat = None     # {pid: khat} from last decision
        self.owner_budget_map = None    # {pid: budget} explicit budgets (optional)
        self.khat_cap = 8.0             # safety cap; see spec I3 / het_point lesson
        self.decision_log = []
        self.rebuild_log = []
        self.log_lock = threading.Lock()
        self._step_time_deque = deque(maxlen=256)
        self._all_step_times = []       # for the warmup base_step_time
        self._base_step_time = None
        self._hits_snapshot = (0, 0)    # (remote_hits, remote_misses) at last boundary

        # health / diagnostic counters (surfaced by the profiler)
        self.stale_window_count = 0
        self.late_window_count = 0
        self.discarded_build_count = 0
        self.peek_clamp_count = 0
        self.fallback_error_count = 0
        self._worker_error = None

    # ------------------------------------------------------------------ #
    # V2 controller plumbing
    # ------------------------------------------------------------------ #
    def set_controller(self, controller):
        """Attach a GreenDyGNNController (spec I2). May be None."""
        self.controller = controller

    def note_step_time(self, st):
        """Trainer feeds per-step wall times (worker uses them for observe())."""
        self._step_time_deque.append(st)
        if len(self._all_step_times) < 2000:
            self._all_step_times.append(st)

    def _get_base_step_time(self):
        if self._base_step_time is None and len(self._all_step_times) >= 50:
            xs = sorted(self._all_step_times[:1000])
            self._base_step_time = xs[max(0, int(0.15 * len(xs)) - 1)]
        return self._base_step_time

    def drain_decision_log(self):
        with self.log_lock:
            out = self.decision_log
            self.decision_log = []
        return out

    def drain_rebuild_log(self):
        with self.log_lock:
            out = self.rebuild_log
            self.rebuild_log = []
        return out

    def get_health_counters(self):
        return {
            "stale_windows": self.stale_window_count,
            "late_windows": self.late_window_count,
            "discarded_builds": self.discarded_build_count,
            "peek_clamps": self.peek_clamp_count,
            "fallback_errors": self.fallback_error_count,
        }

    def _window_hit_rate(self):
        """Remote hit rate over the last window (non-destructive counters)."""
        h, m = self.cache.remote_cache_hits, self.cache.remote_misses
        ph, pm = self._hits_snapshot
        self._hits_snapshot = (h, m)
        dh, dm = h - ph, m - pm
        tot = dh + dm
        return (dh / tot) if tot > 0 else 1.0

    def _run_controller_boundary(self):
        """Observe + decide at a window boundary (worker thread, post-swap)."""
        if self.controller is None:
            return
        try:
            owner_stats = self.cache.get_owner_latency_stats(last_n=512)
            hit_rate = self._window_hit_rate()
            step_time = (sum(self._step_time_deque) / len(self._step_time_deque)
                         if self._step_time_deque else 0.0)
            base = self._get_base_step_time() or step_time
            self.controller.observe(owner_stats, hit_rate, step_time, base)
            t0 = time.perf_counter()
            d = self.controller.decide()
            overhead_ms = (time.perf_counter() - t0) * 1000.0

            if self.rl_enabled and d.W and int(d.W) > 0:
                self.window_size = int(d.W)
            if not self.uniform_alloc:
                # Decision.owner_budgets carries allocation WEIGHTS (capped
                # sigma_hat ratios ~1-8; I2 2026-07-07 redesign), NOT row
                # budgets — route them through the analytic allocator in
                # _select_hot_nodes. None means UNIFORM: the controller only
                # tilts on real per-owner degradation; raw khat (cross-owner
                # rtt/row) is fetch-size-confounded and must never steer the
                # cache.
                wb = getattr(d, "owner_budgets", None)
                if wb:
                    self.controller_khat = {
                        int(p): min(float(k), self.khat_cap)
                        for p, k in dict(wb).items()}
                else:
                    self.controller_khat = None
            else:
                self.controller_khat = None
            self.owner_budget_map = None

            entry = {
                "t": time.time(),
                "batch_idx": self.current_batch_idx,
                "W": int(self.window_size),
                "applied_w": bool(self.rl_enabled),
                "provenance": getattr(d, "provenance", "unknown"),
                "khat": {int(p): float(v) for p, v in dict(getattr(d, "khat", {}) or {}).items()},
                "sigma_hat": {int(p): float(v) for p, v in dict(getattr(d, "sigma_hat", {}) or {}).items()},
                "hit_rate": hit_rate,
                "step_time": step_time,
                "base_step_time": base,
                "overhead_ms": overhead_ms,
            }
            with self.log_lock:
                self.decision_log.append(entry)
        except Exception as e:
            self.fallback_error_count += 1
            print(f"[Prefetcher] controller boundary error: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ #
    # Window schedule (OptiSched legacy)
    # ------------------------------------------------------------------ #
    def set_schedule(self, window_lengths):
        """Install a non-uniform window schedule for the current epoch."""
        if not window_lengths:
            self.schedule = None
            return
        sched, pos = {}, 0
        for w in window_lengths:
            sched[pos] = int(w)
            pos += int(w)
        self.schedule = sched

    def _win_len(self, start_batch_global):
        """Length of the window that starts at a given (1-based global) batch."""
        if not self.schedule:
            return self.window_size
        local = (start_batch_global - 1) % self.batches_per_epoch
        if local in self.schedule:
            return self.schedule[local]
        nxt = min((s for s in self.schedule if s > local), default=None)
        return (nxt - local) if nxt is not None else self.window_size

    def _sanitize_labels(self, labels):
        """Sanitize labels: handle NaN values and clamp to valid range"""
        if labels.is_floating_point():
            labels = th.nan_to_num(labels, nan=-1.0)

        labels = labels.long()
        if self.n_classes is not None:
            labels = th.clamp(labels, min=-1, max=self.n_classes-1)
        else:
            labels = th.clamp(labels, min=-1, max=1000)

        return labels

    # ------------------------------------------------------------------ #
    # Epoch / build lifecycle
    # ------------------------------------------------------------------ #
    def start_epoch(self, epoch):
        self.epoch = epoch

        if epoch == 0:
            self.current_batch_idx = 1
        else:
            self.current_batch_idx = epoch * self.batches_per_epoch + 1

        # Initial cache update (blocking for first window)
        if self.current_batch_idx < self.cache_valid_from_batch or \
           self.current_batch_idx >= self.cache_valid_until_batch:
            self._update_cache_sync(self.current_batch_idx)
            if not self.synchronous_cache and self.cache_builder_thread is not None \
               and self.cache_builder_thread.is_alive():
                # refresh the outstanding request for the new position
                self._request_build(self.cache_valid_until_batch)

        if not self.synchronous_cache:
            self.stop_event.clear()

            # Start worker thread (it exits after batches_per_epoch)
            threading.Thread(target=self._worker, daemon=True).start()

            # Start cache builder thread ONLY if not already running
            if self.cache_builder_thread is None or not self.cache_builder_thread.is_alive():
                self.next_window_ready.clear()
                self.build_next_window.clear()
                self._request_build(self.current_batch_idx + self._win_len(self.current_batch_idx))
                self.cache_builder_thread = threading.Thread(target=self._cache_builder, daemon=True)
                self.cache_builder_thread.start()

    def _request_build(self, start_batch):
        """Issue (or supersede) the outstanding build request."""
        with self.gen_lock:
            self.build_seq += 1
            self._pending_build = (self.build_seq, start_batch)
            self.next_window_ready.clear()
        self.build_next_window.set()

    def _update_cache_sync(self, start_batch):
        """Synchronous cache update (used for first window or sync mode)"""
        win_len, payload = self._build_cache_for_window(start_batch, is_sync=True)
        self._commit_build(payload, expected_active_idx=None)
        self.cache.swap_buffers()
        self.cache_valid_from_batch = start_batch
        self.cache_valid_until_batch = start_batch + win_len

    def _cache_builder(self):
        """Background thread to build cache for the NEXT window"""
        while not self.stop_event.is_set():
            if not self.build_next_window.wait(timeout=0.1):
                continue

            if self.stop_event.is_set():
                break

            self.build_next_window.clear()
            with self.gen_lock:
                if self._pending_build is None:
                    continue
                seq, start_batch = self._pending_build
            active_at_start = self.cache.active_idx

            try:
                win_len, payload = self._build_cache_for_window(start_batch, is_sync=False)
            except Exception as e:
                print(f"[Prefetcher] cache builder error: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                continue

            with self.gen_lock:
                if seq != self.build_seq:
                    # a newer request superseded this build -> discard
                    self.discarded_build_count += 1
                    continue
            committed = self._commit_build(payload, expected_active_idx=active_at_start)
            if not committed:
                self.discarded_build_count += 1
                continue
            with self.gen_lock:
                self.last_built = (seq, start_batch, win_len)
                self.ready_seq = seq
            self.next_window_ready.set()

    def _worker(self):
        # Worker thread for async mode
        try:
            self._worker_body()
        except Exception as e:
            self._worker_error = e
            print(f"[Prefetcher] worker error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            try:
                self.buffer.put(None, timeout=1.0)
            except Exception:
                pass

    def _worker_body(self):
        for step in range(self.batches_per_epoch):
            if self.stop_event.is_set():
                break

            # Check if we crossed window boundary
            if self.current_batch_idx >= self.cache_valid_until_batch:
                with self.gen_lock:
                    expected = self.build_seq
                ok = self.next_window_ready.wait(timeout=self.builder_timeout_s)
                with self.gen_lock:
                    ready_ok = ok and (self.ready_seq == expected)
                    built = self.last_built

                if ready_ok:
                    self.next_window_ready.clear()
                    self.cache.swap_buffers()
                    _seq, built_start, built_len = built
                    if built_start != self.current_batch_idx:
                        self.late_window_count += 1
                    self.cache_valid_from_batch = self.current_batch_idx
                    self.cache_valid_until_batch = self.current_batch_idx + built_len

                    # V2: controller decision for the NEXT window (post-swap)
                    self._run_controller_boundary()

                    self._request_build(self.cache_valid_until_batch)
                else:
                    # Builder late: DO NOT swap (spec P1). Serve the stale but
                    # consistent active buffer and try again next boundary.
                    self.stale_window_count += 1
                    ext = max(1, self._win_len(self.current_batch_idx))
                    print(f"[Prefetcher] Warning: cache builder late at batch "
                          f"{self.current_batch_idx}; serving stale window "
                          f"(+{ext} batches, stale_count={self.stale_window_count})")
                    self.cache_valid_until_batch = self.current_batch_idx + ext

            # Get batch from shared buffer (or initial data)
            batch_data = None
            if self.initial_data_iter:
                try:
                    batch_data = next(self.initial_data_iter)
                except StopIteration:
                    self.initial_data_iter = None

            if batch_data is None:
                batch_data = self.shared_buffer.get()

            if batch_data is None:
                break

            with self.count_lock:
                self.total_popped += 1

            input_nodes, seeds, blocks, remote_mask, labels = batch_data

            feats = self.cache.get_features(input_nodes, self.g, self.device, remote_mask, self.current_batch_idx)
            labels = self._sanitize_labels(labels).to(self.device, non_blocking=True)
            blocks = [b.to(self.device, non_blocking=True) for b in blocks]

            self.buffer.put((feats, labels, blocks))
            self.current_batch_idx += 1

    # ------------------------------------------------------------------ #
    # Cache building
    # ------------------------------------------------------------------ #
    def _select_hot_nodes(self, unique_nodes, node_counts):
        """Top-k hot-node selection with optional owner-aware allocation.

        Priority: controller allocation weights (Decision.owner_budgets =
        capped sigma_hat WEIGHTS, applied via controller_khat through the
        analytic marginal-greedy, spec I2/I3) > legacy owner_kappa
        (optisched) > legacy cost_weights (v1 DQN) > plain top-count.
        NOTE: there is deliberately NO "explicit row budget" path — the
        2026-07-08 matrix collapse came from reading weight values (1-8) as
        row counts, which built a ~empty cache.
        Returns (hot_nodes_cpu LongTensor, budgets dict|None).
        """
        k = min(self.cache.n_hot, unique_nodes.numel())
        if k <= 0:
            return th.tensor([], dtype=th.long), None

        # 1) controller-weight analytic allocation (V2 I2/I3)
        khat = None if self.uniform_alloc else self.controller_khat
        if khat and self.pb is not None:
            try:
                owners_np = self.pb.nid2partid(unique_nodes.cpu()).numpy()
                capped = {int(p): min(float(v), self.khat_cap) for p, v in khat.items()}
                P = self.num_owners or (int(owners_np.max()) + 1)
                if _select_owner_budgets_v2 is not None:
                    sel, budgets = _select_owner_budgets_v2(
                        unique_nodes.cpu().numpy(), node_counts.cpu().numpy(),
                        owners_np, self.cache.n_hot, capped)
                elif select_owner_budgeted is not None:
                    sel, budgets = select_owner_budgeted(
                        unique_nodes.cpu().numpy(), node_counts.cpu().numpy(),
                        owners_np, self.cache.n_hot, capped, P)
                else:
                    raise RuntimeError("no allocator module available")
                return th.from_numpy(sel), budgets
            except Exception as e:
                self.fallback_error_count += 1
                print(f"[Prefetcher] khat allocation failed "
                      f"({type(e).__name__}: {e}); falling back to top-k")

        # 2) legacy optisched owner_kappa
        if self.owner_kappa and self.pb is not None and select_owner_budgeted is not None:
            try:
                owners_np = self.pb.nid2partid(unique_nodes.cpu()).numpy()
                P = self.num_owners or (int(owners_np.max()) + 1)
                sel, budgets = select_owner_budgeted(
                    unique_nodes.cpu().numpy(), node_counts.cpu().numpy(),
                    owners_np, self.cache.n_hot, self.owner_kappa, P)
                return th.from_numpy(sel), budgets
            except Exception as e:
                self.fallback_error_count += 1
                print(f"[Prefetcher] owner_kappa allocation failed "
                      f"({type(e).__name__}: {e}); falling back to top-k")

        # 3) legacy cost-weighted scoring (v1 DQN path)
        if self.cost_weights and self.pb is not None:
            try:
                owners = self.pb.nid2partid(unique_nodes.cpu())
                scores = node_counts.float()
                for pid, weight in self.cost_weights.items():
                    m = (owners == pid)
                    if m.any():
                        scores[m] *= weight
                _, top_indices = th.topk(scores, k)
                return unique_nodes[top_indices], None
            except Exception as e:
                self.fallback_error_count += 1
                print(f"[Prefetcher] cost-weight selection failed "
                      f"({type(e).__name__}: {e}); falling back to top-k")

        # 4) plain top-count
        _, top_indices = th.topk(node_counts, k)
        return unique_nodes[top_indices], None

    def _timed_bulk_pull(self, nodes_cpu):
        """Builder bulk feature pull with the lock/RPC timing split.

        Returns (cpu_feats, lock_s, rpc_s). rpc_s is the production-path RPC
        service time (combined across owners — splitting per owner would
        serialize the pull and change production behavior; per-owner surfaces
        come from microbench_fetch.py instead). The host->device copy is done
        and timed by the CALLER, outside the dist_lock — the lock is no
        longer held during H2D."""
        t0 = time.perf_counter()
        if self.dist_lock:
            with self.dist_lock:
                lock_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                feats = self.g.ndata["features"][nodes_cpu]
                rpc_s = time.perf_counter() - t1
        else:
            lock_s = 0.0
            t1 = time.perf_counter()
            feats = self.g.ndata["features"][nodes_cpu]
            rpc_s = time.perf_counter() - t1
        return feats, lock_s, rpc_s

    def _build_cache_for_window(self, start_batch, is_sync=False):
        """Plan + fetch the cache contents for the window at start_batch.

        Returns (win_len, payload) where payload = (hot_nodes_dev, features)
        ready for _commit_build. Does NOT touch the cache buffers itself
        (the commit is separate so a stale build can be discarded).
        """
        t0 = time.time()
        win_len = self._win_len(start_batch)

        precomputed_hot_nodes = None
        if self.window_hot_nodes and start_batch in self.window_hot_nodes:
             precomputed_hot_nodes = self.window_hot_nodes[start_batch]

        budgets = None
        if precomputed_hot_nodes is not None:
            hot_nodes = precomputed_hot_nodes
            unique_remote_nodes = len(hot_nodes)
            total_accesses = unique_remote_nodes
        else:
            # JIT hot-node calculation using SharedBuffer
            if isinstance(self.shared_buffer, list):
                start_idx = start_batch - 1
                end_idx = start_idx + win_len
                window_items = self.shared_buffer[start_idx:end_idx]
            else:
                window_items = []
                if self.initial_data:
                    curr_batch_idx = start_batch
                    start_list_idx = curr_batch_idx - 1
                    if start_list_idx < len(self.initial_data):
                         end_list_idx = min(start_list_idx + win_len, len(self.initial_data))
                         window_items = self.initial_data[start_list_idx:end_list_idx]

                items_needed = win_len - len(window_items)
                if items_needed > 0:
                    total_initial = len(self.initial_data) if self.initial_data else 0
                    global_idx_needed = (start_batch - 1) + len(window_items)
                    with self.count_lock:
                        popped = self.total_popped
                    effective_head_idx = max(total_initial, popped)
                    buffer_offset = max(0, global_idx_needed - effective_head_idx)
                    # clamp to buffer capacity so peek_n can always return
                    cap = getattr(self.shared_buffer, "capacity", None)
                    want = buffer_offset + items_needed
                    if cap is not None and want > cap:
                        self.peek_clamp_count += 1
                        want = cap
                    items = self.shared_buffer.peek_n(want)
                    window_items.extend(items[buffer_offset : buffer_offset + items_needed])

            all_input_nodes = []
            for data in window_items:
                inp, mask = data[0], data[3]
                if mask.any():
                    all_input_nodes.append(inp[mask])

            unique_remote_nodes = 0
            total_accesses = 0
            hot_nodes = th.tensor([], dtype=th.long)

            if all_input_nodes:
                all_remote = th.cat(all_input_nodes)
                total_accesses = all_remote.numel()
                if total_accesses > 0:
                    unique_nodes, node_counts = th.unique(all_remote, return_counts=True)
                    unique_remote_nodes = unique_nodes.numel()
                    hot_nodes, budgets = self._select_hot_nodes(unique_nodes, node_counts)
            if budgets is not None:
                self.last_owner_budgets = budgets

        t_plan_done = time.time()

        # Phase 2: assemble features (reuse from active buffer + bulk-fetch new)
        active_nodes = self.cache.cache_nodes[self.cache.active_idx]
        active_features = self.cache.cache_features[self.cache.active_idx]

        reused_count = 0
        new_count = 0
        t_lock_s = t_rpc_s = t_h2d_s = 0.0
        pending_features = th.empty((0, self.cache.feat_dim), device=self.device)
        hot_nodes_cpu = th.tensor([], dtype=th.long)
        new_nodes_cpu = th.tensor([], dtype=th.long)

        if len(hot_nodes) > 0:
            hot_nodes_cpu = hot_nodes
            selected_count = len(hot_nodes_cpu)

            if len(active_nodes) > 0:
                old_nodes_cpu = active_nodes.cpu()

                overlap_mask_cpu = th.isin(hot_nodes_cpu, old_nodes_cpu)
                reused_count = int(overlap_mask_cpu.sum())
                new_count = selected_count - reused_count

                # Built off to the side (fresh tensor) BY DESIGN: a swap can
                # land mid-build, and the refusable commit only protects us
                # because a discarded build never touched the live buffers.
                # The caching allocator recycles the freed block, so this is
                # not per-rebuild cudaMalloc churn.
                pending_features = th.empty((selected_count, self.cache.feat_dim),
                                            dtype=active_features.dtype, device=self.device)

                if reused_count > 0:
                    overlap_mask_dev = overlap_mask_cpu.to(self.device)
                    overlap_nodes_dev = hot_nodes_cpu[overlap_mask_cpu].to(self.device)

                    active_idx_map = self.cache.cache_idx[self.cache.active_idx]
                    # .long(): the O(N) idx map is int32; feature indexing
                    # needs long, cast only the small overlap slice.
                    old_positions = active_idx_map[overlap_nodes_dev].long()
                    new_positions = th.nonzero(overlap_mask_dev, as_tuple=True)[0]

                    pending_features[new_positions] = active_features[old_positions]

                    new_nodes_cpu = hot_nodes_cpu[~overlap_mask_cpu]
                    if len(new_nodes_cpu) > 0:
                        new_positions_new = th.nonzero(~overlap_mask_dev, as_tuple=True)[0]
                        pulled, t_lock_s, t_rpc_s = self._timed_bulk_pull(new_nodes_cpu)
                        th2d0 = time.perf_counter()
                        pending_features[new_positions_new] = pulled.to(self.device)
                        t_h2d_s = time.perf_counter() - th2d0
                else:
                    new_nodes_cpu = hot_nodes_cpu
                    pulled, t_lock_s, t_rpc_s = self._timed_bulk_pull(hot_nodes_cpu)
                    th2d0 = time.perf_counter()
                    pending_features = pulled.to(self.device)
                    t_h2d_s = time.perf_counter() - th2d0
            else:
                reused_count = 0
                new_count = selected_count
                new_nodes_cpu = hot_nodes_cpu
                pulled, t_lock_s, t_rpc_s = self._timed_bulk_pull(hot_nodes_cpu)
                th2d0 = time.perf_counter()
                pending_features = pulled.to(self.device)
                t_h2d_s = time.perf_counter() - th2d0

        t_fetch_done = time.time()

        # Rebuild instrumentation (spec I4): rows/bytes actually moved.
        elem = getattr(self.cache, "_elem_size", 4)
        per_owner = {}
        if len(new_nodes_cpu) > 0:
            owners = self.cache.owner_of(new_nodes_cpu)
            if owners is not None:
                for pid in owners.unique().tolist():
                    rows = int((owners == pid).sum())
                    per_owner[int(pid)] = {
                        "rows": rows, "bytes": rows * self.cache.feat_dim * elem}
        entry = {
            "t": t0,
            "start_batch": int(start_batch),
            "win_len": int(win_len),
            "sync": bool(is_sync),
            "t_plan_s": round(t_plan_done - t0, 6),
            "t_fetch_s": round(t_fetch_done - t_plan_done, 6),
            # Blocker-1 split (RESEARCH_PLAN_v2): lock wait / RPC service /
            # host->device copy, separated so the transition model g(...) is
            # not confounded by lock contention. t_fetch_s stays the total
            # assembly phase for back-compat.
            "t_lock_s": round(t_lock_s, 6),
            "t_rpc_s": round(t_rpc_s, 6),
            "t_h2d_s": round(t_h2d_s, 6),
            "unique_remote_nodes": int(unique_remote_nodes),
            "total_remote_accesses": int(total_accesses),
            "rows_reused": int(reused_count),
            "rows_fetched": int(new_count),
            "bytes_fetched": int(new_count) * self.cache.feat_dim * elem,
            "per_owner": per_owner,
            "owner_budgets": {int(k): int(v) for k, v in (budgets or {}).items()},
        }
        with self.log_lock:
            self.rebuild_log.append(entry)

        payload = (hot_nodes_cpu, pending_features)
        return win_len, payload

    def _commit_build(self, payload, expected_active_idx=None):
        """Write the built window into the pending buffer (refusable)."""
        hot_nodes_cpu, pending_features = payload
        if len(hot_nodes_cpu) > 0:
            return self.cache.set_write_buffer_state(
                hot_nodes_cpu.to(self.device), pending_features,
                expected_active_idx=expected_active_idx)
        empty_nodes = th.tensor([], dtype=th.long, device=self.device)
        empty_feats = th.empty((0, self.cache.feat_dim), device=self.device)
        return self.cache.set_write_buffer_state(
            empty_nodes, empty_feats, expected_active_idx=expected_active_idx)

    # ------------------------------------------------------------------ #
    # Consumption
    # ------------------------------------------------------------------ #
    def get(self):
        if not self.synchronous_cache:
            item = self.buffer.get()
            if item is None and self._worker_error is not None:
                raise RuntimeError("Prefetcher worker thread died") from self._worker_error
            return item
        else:
            # Synchronous mode
            if isinstance(self.shared_buffer, list):
                 return None

            if self.initial_data_iter:
                try:
                    batch_data = next(self.initial_data_iter)
                except StopIteration:
                    self.initial_data_iter = None
                    batch_data = self.shared_buffer.get()
            else:
                batch_data = self.shared_buffer.get()

            if batch_data is None:
                raise StopIteration("No more batches")

            with self.count_lock:
                self.total_popped += 1

            input_nodes, seeds, blocks, remote_mask, labels = batch_data

            if self.current_batch_idx >= self.cache_valid_until_batch:
                self._update_cache_sync(self.current_batch_idx)

            feats = self.cache.get_features(input_nodes, self.g, self.device, remote_mask, self.current_batch_idx)
            labels = self._sanitize_labels(labels).to(self.device, non_blocking=True)
            blocks = [b.to(self.device, non_blocking=True) for b in blocks]

            self.current_batch_idx += 1
            return (feats, labels, blocks)

    def get_batch(self, batch_id):
        """Synchronous batch retrieval for autotuning"""
        if isinstance(self.shared_buffer, list):
             idx = batch_id - 1
             if 0 <= idx < len(self.shared_buffer):
                 return self.shared_buffer[idx]
             return None
        else:
             return None

    def stop(self):
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
