"""dgl-free unit/integration tests for the V2 prefetch pipeline (spec P10).

Mocks stand in for the DistGraph / partition book / DistTensor so these run
with plain pytest + torch-cpu + numpy anywhere.
"""

import os
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache import FeatureCache
from prefetcher import BatchPrefetcher, SharedBuffer
from helpers import set_all_seeds, sanitize_labels


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #
class MockDistTensor:
    """numpy/torch-backed stand-in for a DGL DistTensor."""

    def __init__(self, data, delay_s=0.0, delay_per_owner=None, owner_fn=None):
        self.data = data
        self.delay_s = delay_s
        self.delay_per_owner = delay_per_owner or {}
        self.owner_fn = owner_fn

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, nodes):
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.delay_per_owner and self.owner_fn is not None:
            owners = self.owner_fn(nodes)
            for pid, d in self.delay_per_owner.items():
                if bool((owners == pid).any()):
                    time.sleep(d)
        return self.data[nodes]


class MockGraph:
    def __init__(self, num_nodes=200, feat_dim=3, rank=0, pb=None,
                 fetch_delay_s=0.0, delay_per_owner=None, owner_fn=None):
        feats = th.arange(num_nodes, dtype=th.float32).unsqueeze(1).repeat(1, feat_dim)
        labels = th.randint(0, 5, (num_nodes,)).float()
        self._num_nodes = num_nodes
        self.ndata = {
            "features": MockDistTensor(feats, delay_s=fetch_delay_s,
                                       delay_per_owner=delay_per_owner,
                                       owner_fn=owner_fn),
            "labels": MockDistTensor(labels),
        }
        self._rank = rank
        self._pb = pb

    def num_nodes(self):
        return self._num_nodes

    def rank(self):
        return self._rank

    def get_partition_book(self):
        if self._pb is None:
            raise AttributeError("no partition book in mock")
        return self._pb


class MockPB:
    """Partition book: owner = node_id % P."""

    def __init__(self, P=4, local=0):
        self.P = P
        self.partid = local

    def nid2partid(self, nodes):
        return (nodes % self.P).long()

    def num_partitions(self):
        return self.P


class FakeBlock:
    def to(self, device, non_blocking=False):
        return self


def make_batches(num_batches, batch_nodes, num_nodes, seed, local_mod=(4, 0)):
    """Batches of (input_nodes, seeds, blocks, remote_mask, labels)."""
    rng = np.random.RandomState(seed)
    P, local = local_mod
    out = []
    for _ in range(num_batches):
        ids = th.from_numpy(
            rng.choice(num_nodes, size=batch_nodes, replace=False)).long()
        remote_mask = (ids % P) != local
        labels = th.randint(0, 5, (batch_nodes,)).float()
        out.append((ids, ids[:4], [FakeBlock()], remote_mask, labels))
    return out


# --------------------------------------------------------------------------- #
# SharedBuffer
# --------------------------------------------------------------------------- #
def test_shared_buffer_fifo_under_threads():
    buf = SharedBuffer(capacity=16)
    N = 200
    got = []

    def producer():
        for i in range(N):
            buf.put(i)
        buf.mark_finished()

    def consumer():
        while True:
            item = buf.get()
            if item is None:
                break
            got.append(item)

    tp, tc = threading.Thread(target=producer), threading.Thread(target=consumer)
    tp.start(); tc.start()
    tp.join(timeout=10); tc.join(timeout=10)
    assert got == list(range(N))


def test_shared_buffer_peek_does_not_consume():
    buf = SharedBuffer(capacity=32)
    for i in range(10):
        buf.put(i)
    assert buf.peek_n(5) == [0, 1, 2, 3, 4]
    assert buf.size() == 10
    assert buf.get() == 0  # peek left items intact


def test_peek_n_clamped_to_capacity_no_deadlock():
    buf = SharedBuffer(capacity=8)
    for i in range(8):
        buf.put(i)  # buffer full: 8 items
    t0 = time.time()
    items = buf.peek_n(50)  # would deadlock without the capacity clamp
    assert time.time() - t0 < 2.0
    assert items == list(range(8))


def test_peek_n_returns_partial_when_finished():
    buf = SharedBuffer(capacity=32)
    for i in range(3):
        buf.put(i)
    buf.mark_finished()
    assert buf.peek_n(10) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# FeatureCache: swap protocol + owner-attributed fetch events
# --------------------------------------------------------------------------- #
def test_stale_commit_refused_after_swap():
    g = MockGraph(num_nodes=64, feat_dim=2)
    cache = FeatureCache(g, n_hot=8, device="cpu")

    captured_active = cache.active_idx        # builder starts here
    cache.swap_buffers()                      # worker swaps meanwhile

    nodes = th.arange(4)
    feats = th.zeros((4, 2))
    ok = cache.set_write_buffer_state(nodes, feats,
                                      expected_active_idx=captured_active)
    assert ok is False                        # stale build refused
    write_nodes, _, _ = cache.get_write_buffer_indices()
    assert write_nodes.numel() == 0           # pending buffer untouched

    ok2 = cache.set_write_buffer_state(nodes, feats,
                                       expected_active_idx=cache.active_idx)
    assert ok2 is True
    write_nodes, _, idx_map = cache.get_write_buffer_indices()
    assert write_nodes.numel() == 4
    assert int(idx_map[nodes[2]]) == 2


def test_commit_without_expectation_always_lands():
    g = MockGraph(num_nodes=64, feat_dim=2)
    cache = FeatureCache(g, n_hot=8, device="cpu")
    ok = cache.set_write_buffer_state(th.arange(3), th.ones((3, 2)),
                                      expected_active_idx=None)
    assert ok is True


def test_fetch_event_owner_attribution():
    P, local = 4, 0
    pb = MockPB(P=P, local=local)
    g = MockGraph(num_nodes=100, feat_dim=3, rank=local, pb=pb)
    cache = FeatureCache(g, n_hot=8, device="cpu")

    ids = th.arange(1, 13)                    # owners 1,2,3,0,1,2,3,0,...
    remote_mask = (ids % P) != local
    out = cache.get_features(ids, g, "cpu", remote_mask, batch_idx=1)

    # feature correctness: row i == node id (mock features)
    assert th.allclose(out[:, 0], ids.float())

    events = cache.snapshot_fetch_events()
    owners_seen = sorted({e[1] for e in events})
    assert owners_seen == [1, 2, 3]           # local owner 0 NOT recorded
    rows_by_owner = {}
    for (_t, pid, rows, nbytes, rtt, lock_s) in events:
        rows_by_owner[pid] = rows_by_owner.get(pid, 0) + rows
        assert nbytes == rows * 3 * 4         # feat_dim=3 float32
        assert rtt >= 0.0
        assert lock_s >= 0.0                  # lock wait split from wire time
    assert rows_by_owner == {1: 3, 2: 3, 3: 3}

    stats = cache.get_owner_latency_stats()
    assert set(stats.keys()) == {1, 2, 3}
    for s in stats.values():
        assert s["rows"] == 3 and s["n"] == 1
        assert s["mean_rtt"] >= 0.0

    ops, rows, nbytes = cache.get_remote_fetch_counters()
    assert ops == 3 and rows == 9 and nbytes == 9 * 3 * 4


def test_owner_latency_reflects_injected_slowness():
    """A slow owner must show a larger mean_rtt — the I1 congestion signal."""
    P, local = 4, 0
    owner_fn = lambda nodes: (nodes % P).long()
    pb = MockPB(P=P, local=local)
    g = MockGraph(num_nodes=100, feat_dim=3, rank=local, pb=pb,
                  delay_per_owner={3: 0.05}, owner_fn=owner_fn)
    cache = FeatureCache(g, n_hot=8, device="cpu")

    ids = th.arange(1, 41)
    remote_mask = (ids % P) != local
    cache.get_features(ids, g, "cpu", remote_mask, batch_idx=1)

    stats = cache.get_owner_latency_stats()
    assert stats[3]["mean_rtt"] > 5 * max(stats[1]["mean_rtt"],
                                          stats[2]["mean_rtt"])


# --------------------------------------------------------------------------- #
# Prefetcher integration: slow builder never corrupts served features
# --------------------------------------------------------------------------- #
class SlowBuildPrefetcher(BatchPrefetcher):
    build_delay_s = 0.25

    def _build_cache_for_window(self, start_batch, is_sync=False):
        if not is_sync:
            time.sleep(self.build_delay_s)
        return super()._build_cache_for_window(start_batch, is_sync=is_sync)


def _run_pipeline(pf_cls, num_batches=24, window=4, controller=None,
                  rl_enabled=True, uniform_alloc=False, builder_timeout=0.05):
    num_nodes = 200
    g = MockGraph(num_nodes=num_nodes, feat_dim=3, rank=0)
    cache = FeatureCache(g, n_hot=50, device="cpu",
                         owner_of=lambda n: (n % 4).long())
    batches = make_batches(num_batches, batch_nodes=20, num_nodes=num_nodes,
                           seed=7)
    sbuf = SharedBuffer(capacity=500)
    for b in batches:
        sbuf.put(b)
    sbuf.mark_finished()

    pf = pf_cls(g, "cpu", cache, window, sbuf,
                total_batches=num_batches, batches_per_epoch=num_batches,
                max_batches=8, synchronous_cache=False, n_classes=5,
                window_hot_nodes={}, dist_lock=None, initial_data=None)
    pf.builder_timeout_s = builder_timeout
    pf.rl_enabled = rl_enabled
    pf.uniform_alloc = uniform_alloc
    if controller is not None:
        pf.set_controller(controller)

    pf.start_epoch(0)
    served = []
    for _ in range(num_batches):
        item = pf.get()
        assert item is not None
        served.append(item)
        pf.note_step_time(0.01)
    pf.stop()
    return pf, batches, served


def test_slow_builder_serves_correct_features_and_counts_stale():
    pf, batches, served = _run_pipeline(SlowBuildPrefetcher)
    # Feature correctness for EVERY served batch: row value == node id.
    for (ids, _s, _b, _m, _l), (feats, labels, blocks) in zip(batches, served):
        assert th.allclose(feats[:, 0], ids.float()), \
            "cache served corrupted features"
        assert labels.dtype == th.long
    # The slow builder must have exercised the stale-window path,
    # and no build may ever have corrupted the active buffer.
    health = pf.get_health_counters()
    assert health["stale_windows"] > 0
    assert pf._worker_error is None


def test_fast_builder_normal_path():
    pf, batches, served = _run_pipeline(BatchPrefetcher, builder_timeout=10.0)
    for (ids, _s, _b, _m, _l), (feats, _labels, _blocks) in zip(batches, served):
        assert th.allclose(feats[:, 0], ids.float())
    assert pf.get_health_counters()["stale_windows"] == 0


def test_controller_invoked_at_boundaries_and_w_applied():
    calls = {"observe": 0, "decide": 0}

    class FakeController:
        def observe(self, owner_stats, hit_rate, step_time, base_step_time,
                    energy_j=None):
            calls["observe"] += 1

        def decide(self):
            calls["decide"] += 1
            return types.SimpleNamespace(
                W=8, owner_budgets=None, khat={1: 2.0, 2: 1.0, 3: 20.0},
                sigma_hat={1: 1.0}, provenance="dqn")

    pf, _b, _s = _run_pipeline(BatchPrefetcher, controller=FakeController(),
                               builder_timeout=10.0)
    assert calls["decide"] >= 1
    decisions = pf.drain_decision_log()
    assert len(decisions) == calls["decide"]
    assert pf.window_size == 8                        # d.W applied
    # owner_budgets=None means UNIFORM: raw khat is fetch-size-confounded
    # and must never steer allocation (2026-07-07 redesign; the 2026-07-08
    # matrix collapse came from tilting on khat here).
    assert pf.controller_khat is None
    assert pf.owner_budget_map is None
    assert decisions[0]["provenance"] == "dqn"
    rebuilds = pf.drain_rebuild_log()
    assert len(rebuilds) >= 1
    for r in rebuilds:
        assert "rows_fetched" in r and "bytes_fetched" in r and "t_fetch_s" in r


def test_alloc_weights_routed_and_cache_stays_full():
    """Regression for the 2026-07-08 matrix collapse: Decision.owner_budgets
    carries allocation WEIGHTS (capped sigma_hat ratios, ~1-8). They must be
    routed through the analytic allocator — never read as absolute row
    budgets (that built a ~8-row cache, hit 0.985 -> 0.0002) — and the
    weighted selection must still FILL the cache."""
    class WeightsController:
        def observe(self, *a, **k):
            pass

        def decide(self):
            return types.SimpleNamespace(
                W=4, owner_budgets={0: 1.0, 1: 2.0, 2: 1.0, 3: 12.0},
                khat={1: 25.0}, sigma_hat={3: 12.0}, provenance="dqn")

    pf, _b, _s = _run_pipeline(BatchPrefetcher, controller=WeightsController(),
                               builder_timeout=10.0)
    assert pf.owner_budget_map is None                # row-budget path is dead
    assert pf.controller_khat == {0: 1.0, 1: 2.0, 2: 1.0, 3: pf.khat_cap}
    # Selection contract: weighted allocation fills the cache (n_hot=50).
    # The shared harness has no partition book (owner-aware paths disabled
    # there — the very blind spot that let the row-budget bug ship), so
    # attach one for the selection check.
    pf.pb = MockPB(4)
    uniq = th.arange(120, dtype=th.long)
    counts = th.randint(1, 10, (120,))
    hot, budgets = pf._select_hot_nodes(uniq, counts)
    assert hot.numel() == min(pf.cache.n_hot, 120)
    assert budgets is not None and sum(budgets.values()) == hot.numel()


def test_static_w_ablation_ignores_dqn_w():
    class FakeController:
        def observe(self, *a, **k):
            pass

        def decide(self):
            return types.SimpleNamespace(W=128, owner_budgets=None,
                                         khat={}, sigma_hat={},
                                         provenance="dqn")

    pf, _b, _s = _run_pipeline(BatchPrefetcher, controller=FakeController(),
                               rl_enabled=False, builder_timeout=10.0)
    assert pf.window_size == 4                        # unchanged (static)


def test_uniform_alloc_ablation_ignores_khat():
    class FakeController:
        def observe(self, *a, **k):
            pass

        def decide(self):
            return types.SimpleNamespace(W=4, owner_budgets={1: 10},
                                         khat={1: 5.0}, sigma_hat={},
                                         provenance="dqn")

    pf, _b, _s = _run_pipeline(BatchPrefetcher, controller=FakeController(),
                               uniform_alloc=True, builder_timeout=10.0)
    assert pf.controller_khat is None
    assert pf.owner_budget_map is None


def test_worker_error_propagates_to_get():
    g = MockGraph(num_nodes=50, feat_dim=3)
    cache = FeatureCache(g, n_hot=8, device="cpu")

    def boom(*a, **k):
        raise RuntimeError("injected fetch failure")
    cache.get_features = boom

    sbuf = SharedBuffer(capacity=16)
    for b in make_batches(4, 10, 50, seed=1):
        sbuf.put(b)
    sbuf.mark_finished()
    pf = BatchPrefetcher(g, "cpu", cache, 2, sbuf, 4, 4,
                         max_batches=4, synchronous_cache=False, n_classes=5)
    with pytest.raises(RuntimeError):
        pf.start_epoch(0)          # sync first build may already raise
        for _ in range(4):
            item = pf.get()
            if item is None:
                raise RuntimeError("worker died")
    pf.stop()


# --------------------------------------------------------------------------- #
# Seeding + label sanitation
# --------------------------------------------------------------------------- #
def test_seeding_determinism():
    set_all_seeds(42)
    a1, b1 = th.randperm(1000), np.random.rand(50)
    set_all_seeds(42)
    a2, b2 = th.randperm(1000), np.random.rand(50)
    assert th.equal(a1, a2)
    assert np.allclose(b1, b2)
    g1 = set_all_seeds(7)
    g2 = set_all_seeds(7)
    assert th.equal(th.randperm(64, generator=g1), th.randperm(64, generator=g2))


def test_sanitize_labels_nan_and_range():
    labs = th.tensor([0.0, float("nan"), 3.0, 99.0, -5.0])
    out = sanitize_labels(labs, n_classes=5)
    assert out.dtype == th.long
    assert out[1] == -1                    # NaN -> ignore_index
    assert out[3] == 4                     # clamped to n_classes-1
    assert out[4] == -1                    # negative -> ignore_index


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
