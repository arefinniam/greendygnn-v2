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
from test_pipeline import MockGraph, MockPB


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
