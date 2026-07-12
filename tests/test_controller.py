"""Tests for the v2 controller + analytic allocator (V2_SPEC Agent S item 6)."""
import os
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import W_CHOICES, state_dim, STATE_SPEC_VERSION
from greendygnn_agent import (GreenDyGNNController, QNetwork, save_checkpoint,
                              load_checkpoint, Decision)
import alloc
from optisched.live_alloc import select_owner_budgeted


P, LOCAL = 4, 0
REMOTE = [1, 2, 3]


def stats(rtt_per_row={1: 1e-5, 2: 1e-5, 3: 1e-5}, rows=200, n=5):
    """Synthesize I1 owner_stats: {pid: (n, rows, bytes, mean_rtt, rtt_per_row)}."""
    out = {}
    for pid, rpr in rtt_per_row.items():
        mean_rtt = rpr * rows
        out[pid] = (n, rows, rows * 2408, mean_rtt, rpr)
    return out


def make_controller(ckpt=None, **kw):
    return GreenDyGNNController(P, LOCAL, checkpoint_path=ckpt, seed=0, **kw)


def make_checkpoint(tmp_path):
    sdim = state_dim(P, len(W_CHOICES))
    net = QNetwork(sdim, len(W_CHOICES))
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, net, {"state_dim": sdim,
                                "num_actions": len(W_CHOICES),
                                "hidden": 256, "w_choices": list(W_CHOICES)})
    return path, net


# ---------------------------------------------------------------- checkpointing
def test_checkpoint_roundtrip(tmp_path):
    path, net = make_checkpoint(tmp_path)
    net2, cfg = load_checkpoint(path)
    for p1, p2 in zip(net.parameters(), net2.parameters()):
        assert torch.equal(p1, p2)
    assert cfg["w_choices"] == list(W_CHOICES)


def test_checkpoint_version_check(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["state_spec_version"] = STATE_SPEC_VERSION + 1
    bad = str(tmp_path / "bad.pt")
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="state_spec_version"):
        load_checkpoint(bad)


def test_controller_dqn_provenance(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    c = make_controller(path)
    c.observe(stats(), hit_rate=0.8, step_time=0.02, base_step_time=0.02)
    c.finish_warmup()
    d = c.decide()
    assert d.provenance == "dqn"
    assert d.W in W_CHOICES


# ----------------------------------------------------------------- fallback path
def test_fallback_without_checkpoint(capsys):
    c = make_controller(None)
    err = capsys.readouterr().err
    assert "HEURISTIC FALLBACK" in err
    c.observe(stats(), 0.8, 0.02, 0.02)
    c.finish_warmup()
    d = c.decide()
    assert d.provenance == "fallback-heuristic"
    assert d.W in W_CHOICES


def test_online_mode_retired():
    with pytest.raises(NotImplementedError):
        GreenDyGNNController(P, LOCAL, mode="online")


def test_warmup_provenance():
    c = make_controller(None)
    d = c.decide()   # no observations at all -> still warming up
    assert d.provenance == "warmup"
    assert d.owner_budgets is None


# ------------------------------------------------------------------- estimators
def test_khat_tracks_relative_per_row_cost():
    c = make_controller(None)
    for _ in range(30):
        c.observe(stats({1: 1e-5, 2: 1e-5, 3: 5e-5}), 0.8, 0.02, 0.02)
    kh = c.khat()
    assert abs(kh[1] - 1.0) < 0.05 and abs(kh[2] - 1.0) < 0.05
    assert abs(kh[3] - 5.0) < 0.5
    # floors at 1.0
    assert min(kh.values()) >= 1.0


def test_sigma_uses_warmup_baseline():
    c = make_controller(None)
    for _ in range(20):
        c.observe(stats({1: 1e-5, 2: 1e-5, 3: 1e-5}), 0.8, 0.02, 0.02)
    c.finish_warmup()
    for _ in range(40):
        c.observe(stats({1: 1e-5, 2: 1e-5, 3: 3e-5}), 0.8, 0.02, 0.02)
    sg = c.sigma_hat()
    assert sg[3] > 2.0, f"owner 3 rtt tripled, sigma_hat={sg[3]}"
    assert sg[1] < 1.3 and sg[2] < 1.3


def test_missing_owner_never_shifts_slots():
    """v1 bug regression: an owner with zero fetches in a window must keep its
    old estimate and MUST NOT shift other owners' attribution (pid-keyed)."""
    c = make_controller(None)
    for _ in range(20):
        c.observe(stats({1: 1e-5, 2: 1e-5, 3: 6e-5}), 0.8, 0.02, 0.02)
    kh_before = c.khat()
    # owner 2 absent this window (zero misses)
    for _ in range(5):
        c.observe(stats({1: 1e-5, 3: 6e-5}), 0.8, 0.02, 0.02)
    kh = c.khat()
    assert abs(kh[3] - kh_before[3]) < 0.5     # still the slow one
    assert np.argmax([kh[p] for p in REMOTE]) == REMOTE.index(3)


def test_state_vector_shape_bounds():
    c = make_controller(None)
    for _ in range(10):
        c.observe(stats({1: 1e-5, 2: 2e-5, 3: 9e-5}), 0.7, 0.03, 0.02,
                  rebuild_frac=0.2, batches_remaining_norm=0.5)
    s = c.state_vector()
    assert s.shape == (state_dim(P, len(W_CHOICES)),)
    assert np.all(np.isfinite(s)) and np.all(s >= 0) and np.all(s <= 1.0 + 1e-6)


# ---------------------------------------------------------------- decision logic
def test_alloc_weights_capped_and_uniform_when_flat(tmp_path):
    """Allocation keys on sigma_hat = per-owner RTT DEGRADATION vs that
    owner's own warm-up baseline (2026-07-07 redesign): a static per-row cost
    difference must NOT tilt the cache (that is fetch-size structure, not
    congestion); a post-warm-up RTT jump on one owner must, capped."""
    path, _ = make_checkpoint(tmp_path)
    c = make_controller(path)
    # clean warm-up: flat costs freeze the baselines
    for _ in range(30):
        c.observe(stats(), 0.8, 0.02, 0.02)
    c.finish_warmup()
    # owner 3's link degrades 100x after warm-up -> capped budgets
    for _ in range(30):
        c.observe(stats({1: 1e-5, 2: 1e-5, 3: 100e-5}), 0.8, 0.02, 0.02)
    d = c.decide()
    assert d.owner_budgets is not None
    assert d.owner_budgets[3] == pytest.approx(alloc.DEFAULT_KHAT_CAP)  # capped
    assert d.owner_budgets[1] == pytest.approx(1.0, abs=0.1)
    # static per-row asymmetry present from the start (baseline absorbs it)
    # -> NO tilt: this is exactly the owner-1 false-positive seen live
    c1 = make_controller(path)
    for _ in range(30):
        c1.observe(stats({1: 8e-5, 2: 1e-5, 3: 1e-5}), 0.8, 0.02, 0.02)
    c1.finish_warmup()
    for _ in range(10):
        c1.observe(stats({1: 8e-5, 2: 1e-5, 3: 1e-5}), 0.8, 0.02, 0.02)
    assert c1.decide().owner_budgets is None
    # flat costs -> no weights (uniform)
    c2 = make_controller(path)
    for _ in range(30):
        c2.observe(stats(), 0.8, 0.02, 0.02)
    c2.finish_warmup()
    assert c2.decide().owner_budgets is None


def test_uniform_alloc_ablation_flag(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    c = make_controller(path, uniform_alloc=True)
    for _ in range(30):
        c.observe(stats({1: 1e-5, 2: 1e-5, 3: 100e-5}), 0.8, 0.02, 0.02)
    c.finish_warmup()
    assert c.decide().owner_budgets is None


def test_decide_under_5ms(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    c = make_controller(path)
    for _ in range(10):
        c.observe(stats(), 0.8, 0.02, 0.02)
    c.finish_warmup()
    for _ in range(20):
        c.decide()
    oh = c.overhead_ms()
    assert oh["mean_ms"] < 5.0, f"decide() too slow: {oh}"


def test_decision_log_drains(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    c = make_controller(path)
    c.observe(stats(), 0.8, 0.02, 0.02)
    c.finish_warmup()
    c.decide(); c.decide()
    log = c.drain_decision_log()
    assert len(log) == 2 and all("W" in e and "provenance" in e for e in log)
    assert c.drain_decision_log() == []


def test_deploy_mode_never_learns(tmp_path):
    path, net = make_checkpoint(tmp_path)
    c = make_controller(path)
    before = [p.clone() for p in c.q_net.parameters()]
    for _ in range(30):
        c.observe(stats({1: 1e-5, 2: 3e-5, 3: 8e-5}), 0.6, 0.05, 0.02)
        c.finish_warmup()
        c.decide()
    for p0, p1 in zip(before, c.q_net.parameters()):
        assert torch.equal(p0, p1), "deployed policy parameters changed!"


def test_decisions_deterministic_given_signals(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    def run():
        c = make_controller(path)
        ws = []
        for i in range(25):
            c.observe(stats({1: 1e-5, 2: 1e-5, 3: (1 + i % 7) * 1e-5}),
                      0.8, 0.02, 0.02)
            if i == 10:
                c.finish_warmup()
            ws.append(c.decide().W)
        return ws
    assert run() == run()


# -------------------------------------------------------------------- allocator
def rand_instance(rng, n=5000, P_=4):
    nodes = rng.choice(10 ** 6, size=n, replace=False)
    counts = rng.integers(1, 50, size=n)
    owners = rng.integers(0, P_, size=n)
    return nodes, counts.astype(np.int64), owners.astype(np.int64)


def test_allocator_equivalence_with_live_alloc():
    rng = np.random.default_rng(0)
    for trial in range(10):
        nodes, counts, owners = rand_instance(rng)
        khat = {0: 1.0, 1: float(rng.uniform(1, 6)), 2: 1.0,
                3: float(rng.uniform(1, 6))}
        sel_a, bud_a = alloc.select_owner_budgets(nodes, counts, owners,
                                                  1000, khat, 4)
        sel_b, bud_b = select_owner_budgeted(nodes, counts, owners, 1000,
                                             alloc.cap_khat(khat), 4)
        assert set(sel_a.tolist()) == set(sel_b.tolist())
        assert bud_a == bud_b


def test_allocator_none_is_top_count():
    rng = np.random.default_rng(1)
    nodes, counts, owners = rand_instance(rng)
    sel, _ = alloc.select_owner_budgets(nodes, counts, owners, 500, None, 4)
    thr = np.sort(counts)[-500]
    sel_counts = counts[np.isin(nodes, sel)]
    assert sel_counts.min() >= thr - 1  # ties at the boundary allowed


def test_khat_cap_behaviour():
    rng = np.random.default_rng(2)
    # heavy-tailed (zipf) access counts — the realistic GNN hub distribution,
    # and the case the cap exists for: fast-owner HUB nodes must survive an
    # extreme khat estimate (the het_point kappa=16 hit-rate-collapse lesson).
    n = 5000
    nodes = rng.choice(10 ** 6, size=n, replace=False)
    counts = np.minimum(rng.zipf(1.6, size=n), 10 ** 4).astype(np.int64)
    owners = rng.integers(0, 4, size=n).astype(np.int64)
    huge = {0: 1.0, 1: 1.0, 2: 1.0, 3: 200.0}
    capped = {0: 1.0, 1: 1.0, 2: 1.0, 3: alloc.DEFAULT_KHAT_CAP}
    sel_h, bud_h = alloc.select_owner_budgets(nodes, counts, owners, 800, huge, 4)
    sel_c, bud_c = select_owner_budgeted(nodes, counts, owners, 800, capped, 4)
    assert set(sel_h.tolist()) == set(sel_c.tolist())
    assert bud_h == bud_c
    # capped selection keeps fast-owner hubs; raw khat=200 abandons (most of) them
    _, bud_raw = select_owner_budgeted(nodes, counts, owners, 800, huge, 4)
    fast_capped = sum(v for k, v in bud_h.items() if k != 3)
    fast_raw = sum(v for k, v in bud_raw.items() if k != 3)
    assert fast_capped > 0, "cap must leave budget for fast-owner hubs"
    assert fast_capped > fast_raw, (
        f"cap should retain more fast-owner rows than raw khat "
        f"({fast_capped} vs {fast_raw})")


def test_cap_khat_floors_and_clamps():
    assert alloc.cap_khat(None) is None
    out = alloc.cap_khat({1: 0.5, 2: 3.0, 3: 99.0}, khat_cap=8.0)
    assert out == {1: 1.0, 2: 3.0, 3: 8.0}


def test_observe_accepts_dict_owner_stats_from_live_cache():
    """Integration contract: cache.get_owner_latency_stats emits DICT values
    (with string pids after JSON round-trips); observe() must accept them.
    Regression test for the 2026-07-07 live failure ('<=' str vs int)."""
    from greendygnn_agent import GreenDyGNNController
    c = GreenDyGNNController(num_partitions=4, local_pid=0,
                             checkpoint_path=None, mode="deploy", seed=0)
    dict_stats = {
        1: {"n": 5, "rows": 100, "bytes": 4000,
            "mean_rtt": 0.010, "mean_rtt_per_row": 1e-4},
        "2": {"n": 3, "rows": 50, "bytes": 2000,
              "mean_rtt": 0.020, "mean_rtt_per_row": 4e-4},
        "3": {"n": 0, "rows": 0, "bytes": 0,
              "mean_rtt": 0.0, "mean_rtt_per_row": 0.0},
    }
    for _ in range(3):
        c.observe(dict_stats, hit_rate=0.9, step_time=0.1, base_step_time=0.09)
    d = c.decide()
    assert d.W in (1, 2, 4, 8, 16, 32, 64, 128)
    assert d.provenance in ("dqn", "fallback-heuristic", "warmup")
    assert 2 in d.khat or "2" not in d.khat  # pid-keyed as ints


def test_p8_parametric_end_to_end(tmp_path):
    """Scale-readiness (audit): controller, state, allocator, and simulator are
    P-parametric — exercised at P=8 (7 remote owners), and a P=4 checkpoint
    must hard-fail at P=8 (never silently misbehave)."""
    import numpy as np
    import simulator as sim
    P8 = 8
    # simulator: state dim + one seeded episode at P=8
    assert sim.state_dim(P8) == 3 * (P8 - 1) + 4 + len(sim.W_CHOICES)
    p = sim.CalibParams(P=P8)
    env = sim.GreenDyGNNSim(p, seed=0)
    obs = env.reset()
    assert obs.shape == (sim.state_dim(P8),)
    for _ in range(5):
        obs, r, done, info = env.step(3)
        assert np.isfinite(r)
        if done:
            break
    # controller at P=8 (no checkpoint -> heuristic path) with dict stats
    c = GreenDyGNNController(P8, 0, checkpoint_path=None, seed=0)
    stats8 = {pid: {"n": 4, "rows": 100, "bytes": 4000,
                    "mean_rtt": 0.01, "mean_rtt_per_row": 1e-4}
              for pid in range(1, P8)}
    for _ in range(30):
        c.observe(stats8, 0.8, 0.02, 0.02)
    c.finish_warmup()
    d = c.decide()
    assert d.W in c.w_choices and len(d.khat) == P8 - 1
    # P=4 checkpoint refuses to load at P=8
    path, _ = make_checkpoint(tmp_path)
    import pytest as _pt
    with _pt.raises(Exception):
        GreenDyGNNController(P8, 0, checkpoint_path=path, seed=0)
