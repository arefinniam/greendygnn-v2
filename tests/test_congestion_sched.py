#!/usr/bin/env python3
"""Tests for congestion2 (schedule/exposure/tc-command logic) and
parse_results v2 (round-trip on synthetic fixtures). dgl-free, no network:
only pure functions and tmp-dir fixtures are exercised.

Run:  python3 -m pytest tests/test_congestion_sched.py -q
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from congestion2 import (build_iperf_client_cmd, build_tc_cmd,
                         exposure_fraction, squarewave_schedule,
                         tc_state_is_clean, teardown_cmds)
import parse_results as pr


# ---------------------------------------------------------------------------
# Square-wave schedule + exposure accounting
# ---------------------------------------------------------------------------

def test_squarewave_basic():
    sched = squarewave_schedule(duration_s=600, period_s=120, duty=0.5)
    assert sched == [(0, 60), (120, 180), (240, 300), (360, 420), (480, 540)]


def test_squarewave_duty_quarter():
    sched = squarewave_schedule(duration_s=240, period_s=120, duty=0.25)
    assert sched == [(0, 30), (120, 150)]


def test_squarewave_truncated_final_window():
    sched = squarewave_schedule(duration_s=150, period_s=120, duty=0.5)
    assert sched == [(0, 60), (120, 150)]  # final window clipped at duration


def test_squarewave_phase_shift():
    sched = squarewave_schedule(duration_s=300, period_s=120, duty=0.5,
                                phase_s=30)
    assert sched == [(30, 90), (150, 210), (270, 300)]


def test_squarewave_full_duty_is_steady():
    sched = squarewave_schedule(duration_s=100, period_s=50, duty=1.0)
    assert exposure_fraction(sched, 0, 100) == pytest.approx(1.0)


def test_squarewave_invalid_params():
    with pytest.raises(ValueError):
        squarewave_schedule(100, 0, 0.5)
    with pytest.raises(ValueError):
        squarewave_schedule(100, 120, 0.0)


def test_exposure_fraction_cases():
    sched = [(0, 60), (120, 180)]
    assert exposure_fraction(sched, 0, 240) == pytest.approx(0.5)
    assert exposure_fraction(sched, 0, 60) == pytest.approx(1.0)   # fully on
    assert exposure_fraction(sched, 60, 120) == pytest.approx(0.0)  # fully off
    assert exposure_fraction(sched, 30, 150) == pytest.approx(0.5)  # partial
    assert exposure_fraction(sched, 100, 100) == 0.0                # degenerate


def test_exposure_is_duty_at_multiple_periods():
    for duty in (0.25, 0.5, 0.75):
        sched = squarewave_schedule(1200, 120, duty)
        assert exposure_fraction(sched, 0, 1200) == pytest.approx(duty)


# ---------------------------------------------------------------------------
# tc command construction (string-level)
# ---------------------------------------------------------------------------

def test_c1_tbf_cmd():
    cmd = build_tc_cmd("c1", iface="eno1", rate="200mbit")
    assert "sudo tc qdisc del dev eno1 root" in cmd     # del-then-add
    assert "add dev eno1 root tbf rate 200mbit" in cmd
    assert "sport" not in cmd and "u32" not in cmd       # NEVER port-scoped
    assert "burst" in cmd and "latency" in cmd


def test_c1_requires_rate():
    with pytest.raises(ValueError):
        build_tc_cmd("c1", rate=None)


def test_c4_delay_jitter_loss():
    cmd = build_tc_cmd("c4", iface="eno1", delay_ms=15, jitter_ms=5,
                       loss_pct=1)
    assert "netem" in cmd
    assert "delay 15ms 5ms distribution normal" in cmd
    assert "loss gemodel 1% 25%" in cmd
    assert "sport" not in cmd


def test_c4_delay_only_and_loss_only():
    c_delay = build_tc_cmd("c4", delay_ms=10)
    assert "delay 10ms" in c_delay and "gemodel" not in c_delay
    c_loss = build_tc_cmd("c4", loss_pct=0.5)
    assert "gemodel 0.5%" in c_loss and "delay" not in c_loss
    with pytest.raises(ValueError):
        build_tc_cmd("c4")  # neither


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        build_tc_cmd("c9", rate="1mbit")


def test_iperf_client_direction():
    egress = build_iperf_client_cmd("10.52.3.89", streams=8, duration_s=30,
                                    direction="egress")
    assert " -R" in egress            # victim (server side) transmits
    assert "-P 8" in egress and "-t 30" in egress and "10.52.3.89" in egress
    ingress = build_iperf_client_cmd("10.52.3.89", direction="ingress")
    assert " -R" not in ingress


# ---------------------------------------------------------------------------
# Teardown idempotency + clean-state detection
# ---------------------------------------------------------------------------

def test_teardown_cmds_idempotent_and_safe():
    cmds = teardown_cmds("eno1")
    assert cmds == teardown_cmds("eno1")            # pure / repeatable
    for c in cmds:
        assert "|| true" in c                        # never fails when clean
    assert any("tc qdisc del dev eno1 root" in c for c in cmds)
    assert any("iperf3" in c for c in cmds)          # kills organic load too


def test_tc_state_clean_detection():
    clean = ("qdisc mq 0: root\n"
             "qdisc fq_codel 0: parent :1 limit 10240p flows 1024")
    assert tc_state_is_clean(clean)
    for dirty_kw in ("tbf", "netem", "htb", "prio"):
        assert not tc_state_is_clean(clean + f"\nqdisc {dirty_kw} 8001: root")


# ---------------------------------------------------------------------------
# Journal -> exposure reconstruction (parse_results)
# ---------------------------------------------------------------------------

def test_journal_on_windows_squarewave():
    t0 = 1000.0
    events = [
        {"t": t0, "action": "run_start"},
        {"t": t0 + 5, "action": "wave_on"},
        {"t": t0 + 5.1, "action": "applied"},
        {"t": t0 + 65, "action": "removed"},
        {"t": t0 + 65.1, "action": "wave_off"},
        {"t": t0 + 125, "action": "wave_on"},
        {"t": t0 + 185, "action": "removed"},
        {"t": t0 + 200, "action": "run_end"},
    ]
    win = pr.journal_on_windows(events)
    assert win == [(t0 + 5, t0 + 65), (t0 + 125, t0 + 185)]
    assert pr.realized_exposure(win, t0, t0 + 200) == pytest.approx(0.6)


def test_journal_steady_apply_teardown():
    events = [{"t": 10.0, "action": "applied"},
              {"t": 110.0, "action": "teardown_verified"}]
    win = pr.journal_on_windows(events)
    assert win == [(10.0, 110.0)]
    assert pr.realized_exposure(win, 10.0, 110.0) == pytest.approx(1.0)


def test_per_epoch_exposure_anchoring():
    # profiler ran [t=100, t=200]; impairment ON during [150, 200]
    windows = [(150.0, 200.0)]
    profile = {"total_wall_time_s": 100.0,
               "epochs": [{"epoch": 0, "wall_time_s": 50.0},
                          {"epoch": 1, "wall_time_s": 100.0}]}
    pe = pr.per_epoch_exposure(windows, t_end_run=200.0,
                               part0_profile=profile)
    assert pe[0]["exposure"] == pytest.approx(0.0)   # epoch 0: [100,150] clean
    assert pe[1]["exposure"] == pytest.approx(1.0)   # epoch 1: [150,200] on


# ---------------------------------------------------------------------------
# SUMMARY-line + numeric helpers
# ---------------------------------------------------------------------------

def test_parse_summary_lines():
    text = ("noise\n"
            "SUMMARY method=greendygnn_v2 part=0 gpu_j=180.5 cpu_j=26000.1 "
            "total_j=26180.6 acc=0.916\n"
            "SUMMARY method=greendygnn_v2 part=1 gpu_j=175.0 cpu_j=25000.0 "
            "total_j=25175.0 acc=0.914\n"
            "not a summary SUMMARY=fake\n")
    recs = pr.parse_summary_lines(text)
    assert len(recs) == 2
    assert recs[0]["part"] == 0 and recs[0]["gpu_j"] == pytest.approx(180.5)
    assert recs[1]["method"] == "greendygnn_v2"


def test_mean_std():
    mu, sd, n = pr.mean_std([1.0, 2.0, 3.0])
    assert (mu, n) == (2.0, 3) and sd == pytest.approx(1.0)
    mu, sd, n = pr.mean_std([5.0])
    assert (mu, sd, n) == (5.0, 0.0, 1)
    assert pr.mean_std([])[2] == 0


def test_flatten_numeric_skips_bools_and_nests():
    flat = pr.flatten_numeric({"a": 1, "b": True, "c": {"d": 2.5, "e": "x"}})
    assert flat == {"a": 1.0, "c.d": 2.5}


# ---------------------------------------------------------------------------
# Matrix round-trip on synthetic fixtures
# ---------------------------------------------------------------------------

def _make_run(root, dataset, cond, method, seed, total_j, epoch_t, acc,
              with_journal=False):
    rd = root / "runs" / f"{dataset}__{cond}__{method}__seed{seed}"
    rd.mkdir(parents=True)
    (rd / "p1_aggregate.json").write_text(json.dumps({
        "sys_total_j_per_epoch": total_j, "sys_cpu_j_per_epoch": total_j * 0.9,
        "epoch_time_s": epoch_t, "accuracy": acc}))
    t_launch, t_end = 1000.0, 1000.0 + 60.0
    for part in range(4):
        prof = {
            "method": method, "part_id": part, "total_wall_time_s": 50.0,
            "steps": [{"epoch": 0, "step": s, "cache_hit_pct": 60.0,
                       "remote_bytes": 1e6, "remote_rows": 1000,
                       "remote_fetch_count": 3,
                       "controller_overhead_ms": 0.4} for s in range(4)],
            "epochs": [{"epoch": 0, "epoch_time_s": epoch_t,
                        "wall_time_s": 25.0, "avg_accuracy": acc},
                       {"epoch": 1, "epoch_time_s": epoch_t,
                        "wall_time_s": 50.0, "avg_accuracy": acc}],
        }
        (rd / f"{method}_part{part}_profile.json").write_text(json.dumps(prof))
    (rd / "run.log").write_text(
        "\n".join(f"SUMMARY method={method} part={p} gpu_j=100.0 "
                  f"cpu_j=1000.0 total_j=1100.0" for p in range(4)) + "\n")
    (rd / "run_meta.json").write_text(json.dumps({
        "dataset": dataset, "condition": cond, "method": method,
        "seed": seed, "t_launch": t_launch, "t_end": t_end, "exit_code": 0}))
    if with_journal:
        events = [{"t": t_launch + 5, "action": "applied", "node": "gnn4"},
                  {"t": t_end - 5, "action": "teardown_verified"}]
        (rd / "cong_journal.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n")
    return rd


def test_matrix_round_trip(tmp_path):
    _make_run(tmp_path, "reddit", "c1_200", "greendygnn_v2", 0,
              total_j=1500.0, epoch_t=2.7, acc=0.91, with_journal=True)
    _make_run(tmp_path, "reddit", "c1_200", "greendygnn_v2", 1,
              total_j=1700.0, epoch_t=2.9, acc=0.92, with_journal=True)
    _make_run(tmp_path, "reddit", "c1_200", "static_w16", 0,
              total_j=2600.0, epoch_t=4.2, acc=0.91)

    groups = pr.scan_matrix(tmp_path)
    assert set(groups) == {("reddit", "c1_200", "greendygnn_v2"),
                           ("reddit", "c1_200", "static_w16")}

    table = pr.aggregate_groups(groups)
    v2 = next(r for r in table if r["method"] == "greendygnn_v2")
    m = v2["metrics"]["agg.sys_total_j_per_epoch"]
    assert m["mean"] == pytest.approx(1600.0)
    assert m["std"] == pytest.approx(141.42, rel=1e-3)
    assert m["n"] == 2 and v2["n_seeds"] == 2 and v2["seeds"] == [0, 1]

    # SUMMARY lines: 4 parts x 1100 J
    assert v2["metrics"]["summary.total_j_sum_allparts"]["mean"] == \
        pytest.approx(4400.0)
    # profile-derived: bytes summed across 4 parts x 4 steps x 1e6
    assert v2["metrics"]["prof.remote_bytes_total_allparts"]["mean"] == \
        pytest.approx(16e6)
    # steady journal covers ~[5, 55] of a 60 s run
    assert v2["metrics"]["exposure.run_fraction"]["mean"] == \
        pytest.approx(50.0 / 60.0, rel=1e-3)
    # provenance present for every seed
    assert len(v2["provenance"]) == 2
    assert all("run_dir" in p for p in v2["provenance"])

    md = pr.to_markdown(table)
    assert "greendygnn_v2" in md and "static_w16" in md
    assert "J/ep" in md and "expo" in md


def test_single_run_mode(tmp_path):
    rd = _make_run(tmp_path, "reddit", "clean", "default_dgl", 0,
                   total_j=3400.0, epoch_t=9.2, acc=0.90)
    out = tmp_path / "metrics.json"
    pr.single_run(str(rd / "run.log"), str(out), "default_dgl", "reddit",
                  "2000")
    data = json.loads(out.read_text())
    assert len(data["summaries"]) == 4
    assert data["metrics"]["summary.total_j_sum_allparts"] == \
        pytest.approx(4400.0)
