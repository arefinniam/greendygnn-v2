"""Tests for the RESEARCH_PLAN_v2 item-2/3 instrumentation (2026-07-12):
scripted-W controller, flight recorder, lock/wire split stats key."""

import json
import os
import sys
import time

import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripted_controller import ScriptedWController, W_LADDER
from flight_recorder import FlightRecorder
from cache import FeatureCache
from test_pipeline import MockGraph, MockPB, FakeBlock


# --------------------------- scripted controller -------------------------- #

def test_scripted_explicit_schedule():
    c = ScriptedWController("16@0,128@48,16@200")
    ws, starts = [], []
    for _ in range(12):
        starts.append(c.planned_batches)
        d = c.decide()
        assert d.provenance == "scripted" and d.owner_budgets is None
        ws.append(d.W)
    # W=16 until planned batches reach 48 (3 decisions: 0,16,32),
    # then W=128 until 200, then 16 again.
    assert ws[:3] == [16, 16, 16]
    assert ws[3] == 128                     # decided at planned batch 48
    first_back = next(i for i, s in enumerate(starts) if s >= 200)
    assert ws[first_back] == 16


def test_scripted_random_deterministic_and_switching():
    a = ScriptedWController("random:7:32:64")
    b = ScriptedWController("random:7:32:64")
    seq_a = [a.decide().W for _ in range(50)]
    seq_b = [b.decide().W for _ in range(50)]
    assert seq_a == seq_b                   # same seed -> same schedule
    assert all(w in W_LADDER for w in seq_a)
    assert len(set(seq_a)) > 1              # actually switches
    # every switch is a REAL transition (never re-picks the current W)
    log_ws = [w for (_s, w) in a.schedule_log]
    changes = [(x, y) for x, y in zip(log_ws, log_ws[1:]) if x != y]
    assert all(x != y for x, y in changes)
    c = ScriptedWController("random:8:32:64")
    assert [c.decide().W for _ in range(50)] != seq_a  # seed matters


def test_scripted_rejects_bad_scripts():
    import pytest
    with pytest.raises(ValueError):
        ScriptedWController("128@10,16@0")   # must start at 0
    with pytest.raises(ValueError):
        ScriptedWController("13@0")          # not in ladder
    with pytest.raises(ValueError):
        ScriptedWController("random:1:0:5")  # bad dwell


def test_scripted_controller_drives_prefetcher_interface():
    """observe() then decide() exactly as _run_controller_boundary calls it."""
    c = ScriptedWController("16@0,32@16")
    c.observe({1: {"n": 1, "rows": 5, "bytes": 100, "mean_rtt": 0.01,
                   "mean_rtt_per_row": 1e-4, "mean_lock_s": 0.0}},
              hit_rate=0.8, step_time=0.1, base_step_time=0.09)
    d1, d2 = c.decide(), c.decide()
    assert (d1.W, d2.W) == (16, 32)


# ----------------------------- flight recorder ---------------------------- #

def test_flight_recorder_writes_and_degrades(tmp_path):
    out = str(tmp_path / "flight.jsonl")
    fr = FlightRecorder(out, interval_s=0.2, iface="lo")
    fr.start()
    time.sleep(0.7)
    fr.stop()
    lines = [json.loads(l) for l in open(out)]
    assert lines[0].get("meta") is True
    samples = lines[1:]
    assert len(samples) >= 2
    for s in samples:
        assert "t" in s
    # on 'lo' the NIC counters must be present and monotone
    if lines[0]["nic"]:
        rx = [s["rx"] for s in samples if s.get("rx") is not None]
        assert rx == sorted(rx)


def test_flight_recorder_absent_iface_is_harmless(tmp_path):
    out = str(tmp_path / "flight2.jsonl")
    fr = FlightRecorder(out, interval_s=0.2, iface="definitely_not_a_nic0")
    fr.start()
    time.sleep(0.4)
    fr.stop()
    lines = [json.loads(l) for l in open(out)]
    assert lines[0]["nic"] is False and len(lines) >= 2


# ----------------------------- lock/wire split ---------------------------- #

def test_owner_stats_expose_mean_lock_s():
    pb = MockPB(P=4, local=0)
    g = MockGraph(num_nodes=40, feat_dim=3, rank=0, pb=pb)
    cache = FeatureCache(g, n_hot=8, device="cpu")
    ids = th.arange(1, 9)
    cache.get_features(ids, g, "cpu", (ids % 4) != 0, batch_idx=1)
    stats = cache.get_owner_latency_stats()
    for s in stats.values():
        assert "mean_lock_s" in s and s["mean_lock_s"] >= 0.0
        assert s["mean_rtt"] >= 0.0


def test_lock_wait_actually_separated_from_rtt():
    """A held dist_lock must show up in lock_s, not in rtt."""
    import threading
    pb = MockPB(P=4, local=0)
    g = MockGraph(num_nodes=40, feat_dim=3, rank=0, pb=pb)
    lock = threading.Lock()
    cache = FeatureCache(g, n_hot=8, device="cpu", dist_lock=lock)

    HOLD = 0.15
    holder = threading.Thread(
        target=lambda: (lock.acquire(), time.sleep(HOLD), lock.release()))
    holder.start()
    time.sleep(0.02)                        # ensure holder owns the lock
    ids = th.tensor([1], dtype=th.long)     # owner 1, remote
    cache.get_features(ids, g, "cpu", th.tensor([True]), batch_idx=1)
    holder.join()

    (_t, _pid, _rows, _b, rtt, lock_s) = cache.snapshot_fetch_events()[0]
    assert lock_s >= HOLD * 0.5             # waited on the held lock
    assert rtt < HOLD * 0.5                 # wire time did NOT absorb the wait


# ------------------- blockers round (2026-07-13 review) ------------------- #

def test_trace_digest_equivalence_and_sensitivity():
    from trace_digest import new_digest, update_digest
    a, b = new_digest(), new_digest()
    t1, t2 = th.tensor([5, 9, 13]), th.tensor([7, 2])
    for h in (a, b):
        update_digest(h, 0, t1)
        update_digest(h, 1, t2)
    assert a.hexdigest() == b.hexdigest()          # same stream, same digest
    c = new_digest()
    update_digest(c, 0, t1)
    update_digest(c, 1, th.tensor([2, 7]))         # order matters
    assert c.hexdigest() != a.hexdigest()
    d = new_digest()
    update_digest(d, 1, t1)                        # batch index matters
    update_digest(d, 0, t2)
    assert d.hexdigest() != a.hexdigest()


def test_background_sampler_writes_digest(tmp_path):
    from prefetcher import BackgroundSampler, SharedBuffer
    from trace_digest import new_digest, update_digest

    class TinySampler:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

    g = MockGraph(num_nodes=40, feat_dim=3, rank=0, pb=MockPB(4, 0))
    lmask = th.zeros(40, dtype=th.bool)
    lmask[th.arange(0, 40, 4)] = True              # owner-0 nodes local
    batches = [(th.tensor([1, 4, 6]), th.tensor([1]), [FakeBlock()]),
               (th.tensor([2, 8, 11]), th.tensor([2]), [FakeBlock()])]
    buf = SharedBuffer(capacity=10)
    dpath = str(tmp_path / "digest.json")
    bs = BackgroundSampler(TinySampler(batches), buf, g, lmask,
                           num_epochs=1, digest_path=dpath)
    bs.start(); bs.join(timeout=10)

    rec = json.load(open(dpath))
    ref = new_digest()
    update_digest(ref, 0, th.tensor([1, 6]))       # 4 is local
    update_digest(ref, 1, th.tensor([2, 11]))      # 8 is local
    assert rec["digest"] == ref.hexdigest() and rec["n_batches"] == 2


def test_background_sampler_trace_dump_matches_digest(tmp_path):
    """Live trace dump must be a byte-exact record of the digest stream.

    Distributed sampling is nondeterministic across runs (server-side RNG
    draw order depends on request interleaving — proven by the 2026-07-15
    smoke campaign), so traces are dumped BY the run itself and validated by
    recomputing the rolling digest from the dumped npz files.
    """
    from prefetcher import BackgroundSampler, SharedBuffer
    from trace_digest import digest_from_traces
    import glob

    class TinySampler:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

    g = MockGraph(num_nodes=40, feat_dim=3, rank=0, pb=MockPB(4, 0))
    lmask = th.zeros(40, dtype=th.bool)
    lmask[th.arange(0, 40, 4)] = True              # owner-0 nodes local
    batches = [(th.tensor([1, 4, 6]), th.tensor([1]), [FakeBlock()]),
               (th.tensor([2, 8, 11]), th.tensor([2]), [FakeBlock()])]
    buf = SharedBuffer(capacity=10)
    dpath = str(tmp_path / "digest.json")
    ddir = str(tmp_path / "trace_dump")
    bs = BackgroundSampler(TinySampler(batches), buf, g, lmask,
                           num_epochs=2, digest_path=dpath,
                           trace_dump_dir=ddir)
    bs.start(); bs.join(timeout=10)

    rec = json.load(open(dpath))
    npz = sorted(glob.glob(os.path.join(ddir, "trace_part0_ep*.npz")))
    assert len(npz) == 2                           # one Trace per epoch
    h, nb = digest_from_traces(npz)
    assert h.hexdigest() == rec["digest"] and nb == rec["n_batches"] == 4
    meta = json.load(open(os.path.join(ddir, "trace_part0_meta.json")))
    assert meta["digest"] == rec["digest"]
    assert meta["epochs_dumped"] == 2 and len(meta["files"]) == 2
    # owners recorded via the partition book (owner = id % P in the mock)
    from optisched.trace import Trace
    tr = Trace.load(npz[0])
    nodes, owners = tr.batch(0)
    assert list(nodes) == [1, 6] and list(owners) == [1, 2]


def test_rebuild_log_has_lock_rpc_h2d_split():
    from prefetcher import BatchPrefetcher
    from test_pipeline import _run_pipeline
    pf, _batches, _served = _run_pipeline(BatchPrefetcher)
    rebuilds = pf.drain_rebuild_log()
    assert rebuilds
    for e in rebuilds:
        for k in ("t_lock_s", "t_rpc_s", "t_h2d_s"):
            assert k in e and e[k] >= 0.0
        # split components can never exceed the total assembly phase
        assert e["t_lock_s"] + e["t_rpc_s"] + e["t_h2d_s"] \
            <= e["t_fetch_s"] + 0.05


def test_scripted_controller_through_real_pipeline():
    from prefetcher import BatchPrefetcher
    from test_pipeline import _run_pipeline
    c = ScriptedWController("4@0,8@8")
    pf, _b, served = _run_pipeline(BatchPrefetcher, num_batches=24,
                                   window=4, controller=c, rl_enabled=True)
    assert len(served) == 24
    dec = pf.drain_decision_log()
    assert dec and all(e["provenance"] == "scripted" for e in dec)
    ws = {e["W"] for e in dec}
    assert 8 in ws                                  # the switch was applied


def test_flight_recorder_node_source_arbitration(tmp_path):
    a = FlightRecorder(str(tmp_path / "a.jsonl"), interval_s=0.2, iface="lo")
    b = FlightRecorder(str(tmp_path / "b.jsonl"), interval_s=0.2, iface="lo")
    try:
        assert a.node_sources is True
        assert b.node_sources is False              # second rank: GPU only
        b.start(); time.sleep(0.3); b.stop()
        lines = [json.loads(l) for l in open(str(tmp_path / "b.jsonl"))]
        assert all("rx" not in s and "cpu" not in s and "rapl" not in s
                   for s in lines[1:])
    finally:
        a.stop(); b.stop()
    c = FlightRecorder(str(tmp_path / "c.jsonl"), interval_s=0.2, iface="lo")
    try:
        assert c.node_sources is True               # lock released by a.stop()
    finally:
        c.stop()
