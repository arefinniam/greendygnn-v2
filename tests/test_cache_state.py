"""Regression tests for the FeatureCache efficiency pass (2026-07-11).

Covers the three behavior-preserving optimizations:
  1. O(k) index-map reset (scatter -1 at the previous commit's nodes instead
     of an O(num_nodes) fill) -- stale entries must still always clear.
  2. Direct host->device miss scatter (no CPU staging tensor) -- returned
     features and all hit/miss counters must be exact.
  3. int32 index map -- dtype guard so the O(N) map never silently reverts
     to int64.

dgl-free: reuses the MockGraph/MockPB harness from test_pipeline.
"""

import os
import sys

import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache import FeatureCache
from test_pipeline import MockGraph, MockPB


def _mk_cache(num_nodes=40, feat_dim=3, local=0, P=4):
    pb = MockPB(P=P, local=local)
    g = MockGraph(num_nodes=num_nodes, feat_dim=feat_dim, rank=local, pb=pb,
                  owner_fn=pb.nid2partid)
    cache = FeatureCache(g, n_hot=10, device="cpu")
    return g, cache


def _feats_for(nodes, feat_dim=3):
    # MockGraph features: row i == [i, i, ..., i]
    return nodes.float().unsqueeze(1).repeat(1, feat_dim)


def test_idx_map_dtype_is_int32():
    _, cache = _mk_cache()
    assert cache.cache_idx[0].dtype == th.int32
    assert cache.cache_idx[1].dtype == th.int32


def test_idx_map_stale_entries_cleared_across_commits():
    """Two commits into the same write buffer: entries from the first commit
    that are absent from the second must read -1 (the O(k) reset must be
    equivalent to the old full fill)."""
    _, cache = _mk_cache()
    write_idx = 1 - cache.active_idx
    idx_map = cache.cache_idx[write_idx]

    a = th.tensor([1, 2, 3], dtype=th.long)
    assert cache.set_write_buffer_state(a, _feats_for(a))
    assert idx_map[1] == 0 and idx_map[2] == 1 and idx_map[3] == 2

    b = th.tensor([3, 4], dtype=th.long)
    assert cache.set_write_buffer_state(b, _feats_for(b))
    assert idx_map[1] == -1 and idx_map[2] == -1  # stale entries cleared
    assert idx_map[3] == 0 and idx_map[4] == 1
    assert int((idx_map >= 0).sum()) == 2  # full-map invariant


def test_idx_map_cleared_across_swap_cycles():
    """Recommitting a buffer after it has been active must still clear its
    previous generation's entries."""
    _, cache = _mk_cache()
    a = th.tensor([1, 2, 3], dtype=th.long)
    assert cache.set_write_buffer_state(a, _feats_for(a))
    cache.swap_buffers()  # {1,2,3} now active

    b = th.tensor([5, 6], dtype=th.long)
    assert cache.set_write_buffer_state(b, _feats_for(b))
    cache.swap_buffers()  # {5,6} active, {1,2,3} back to write side

    c = th.tensor([7], dtype=th.long)
    assert cache.set_write_buffer_state(c, _feats_for(c))
    idx_map = cache.cache_idx[1 - cache.active_idx]
    assert idx_map[1] == -1 and idx_map[2] == -1 and idx_map[3] == -1
    assert idx_map[7] == 0
    assert int((idx_map >= 0).sum()) == 1


def test_get_features_hit_miss_content_and_counters():
    """End-to-end through the rewritten hit path + direct miss scatter:
    exact feature content, exact counters, correct per-owner events."""
    g, cache = _mk_cache(num_nodes=40, local=0, P=4)
    cached = th.tensor([5, 9, 13], dtype=th.long)  # owners 1,1,1 (remote)
    assert cache.set_write_buffer_state(cached, _feats_for(cached))
    cache.swap_buffers()

    ids = th.tensor([5, 6, 9, 12, 0], dtype=th.long)
    remote_mask = (ids % 4) != 0  # [T, T, T, F, F]
    out = cache.get_features(ids, g, "cpu", remote_mask, batch_idx=1)

    assert th.equal(out, _feats_for(ids))  # content exact for hits AND misses
    assert cache.cache_hits == 2           # 5, 9
    assert cache.remote_cache_hits == 2
    assert cache.remote_misses == 1        # 6 (owner 2)
    assert cache.local_misses == 2         # 12, 0
    events = cache.snapshot_fetch_events()
    assert len(events) == 1                # one remote owner pulled
    _t, pid, rows, _b, _rtt = events[0]
    assert pid == 2 and rows == 1          # 6 % 4 == 2
    assert cache.get_step_remote_hit_rate() == 2 / 3 * 100


def test_get_features_empty_cache_all_miss():
    g, cache = _mk_cache(num_nodes=40, local=0, P=4)
    ids = th.tensor([1, 2, 8], dtype=th.long)   # owners 1, 2, 0(local)
    remote_mask = (ids % 4) != 0
    out = cache.get_features(ids, g, "cpu", remote_mask, batch_idx=1)
    assert th.equal(out, _feats_for(ids))
    assert cache.cache_hits == 0
    assert cache.remote_misses == 2
    assert cache.local_misses == 1
