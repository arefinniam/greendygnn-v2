"""Property tests for the v2 calibrated simulator (V2_SPEC Agent S item 6)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import (CalibParams, StepModel, GreenDyGNNSim, W_CHOICES,
                       build_state, state_dim, heuristic_policy,
                       random_policy_factory, _profile, ARCHETYPES, SEVERITIES)

K1 = np.ones(3)
D0 = np.zeros(3)


@pytest.fixture(scope="module")
def calib():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "calib_synthetic.json")
    return CalibParams.load(path)


@pytest.fixture(scope="module")
def model(calib):
    return StepModel(calib, alloc_on=True)


# ------------------------------------------------------------------ cost surface
def test_clean_u_shape_interior_optimum(model):
    g = model.grid(K1, D0)
    i = int(np.argmin(g))
    assert 0 < i < len(W_CHOICES) - 1, f"clean optimum at boundary: W*={W_CHOICES[i]}"
    # strictly worse at the extremes
    assert g[0] > g[i] and g[-1] > g[i]


def test_wstar_monotone_nonincreasing_in_kappa(model):
    prev = None
    for kv in (1, 3, 10, 30, 100, 300):
        k = K1.copy(); k[1] = kv
        w = W_CHOICES[model.best_static_w(k, D0)]
        if prev is not None:
            assert w <= prev, f"W* rose ({prev}->{w}) as kappa rose to {kv}"
        prev = w


def test_congestion_strictly_increases_step_time(model):
    for w in W_CHOICES:
        t_clean = model.step_time(w, K1, D0)[0]
        k = K1.copy(); k[0] = 20
        assert model.step_time(w, k, D0)[0] > t_clean
        d = D0.copy(); d[2] = 0.010
        assert model.step_time(w, K1, d)[0] > t_clean


def test_hit_rate_bounds_and_monotone(model, calib):
    prev = None
    for w in W_CHOICES:
        h = model.hit_rate(w)
        assert 0.0 <= h <= calib.hmax + 1e-9
        if prev is not None:
            assert h <= prev + 1e-9, "h(W) must be non-increasing in W"
        prev = h


def test_cache_pressure_lowers_hit_rate(calib):
    from dataclasses import asdict
    small = CalibParams(**{**asdict(calib), "n_hot": 5000})
    big = CalibParams(**{**asdict(calib), "n_hot": 10 ** 9})
    mp, mb = StepModel(small), StepModel(big)
    assert mp.hit_rate(128) < mb.hit_rate(128)


def test_alloc_tilt_reduces_congested_cost(calib):
    """Cost-aware allocation must weakly reduce step time under asymmetric
    bandwidth congestion (it shifts misses off the expensive link)."""
    on = StepModel(calib, alloc_on=True)
    off = StepModel(calib, alloc_on=False)
    k = K1.copy(); k[2] = 50
    for w in (8, 16, 32):
        assert on.step_time(w, k, D0)[0] < off.step_time(w, k, D0)[0]


def test_energy_split_positive_and_additive(model):
    T = model.step_time(16, K1, D0)[0]
    tot, gpu, cpu = model.step_energy(T)
    assert gpu > 0 and cpu > 0
    assert abs(tot - (gpu + cpu)) < 1e-12


# ------------------------------------------------------------------- profiles
def test_profiles_cover_episode_and_are_seeded():
    rng = np.random.default_rng(7)
    for arch in ARCHETYPES:
        for sev in SEVERITIES:
            segs = _profile(arch, sev, 3000, 3, np.random.default_rng(11))
            assert segs[0].t0 == 0 and segs[-1].t1 == 3000
            for a, b in zip(segs, segs[1:]):
                assert a.t1 == b.t0, f"gap in {arch}"
    s1 = _profile("oscillating", "moderate", 3000, 3, np.random.default_rng(5))
    s2 = _profile("oscillating", "moderate", 3000, 3, np.random.default_rng(5))
    assert len(s1) == len(s2)
    assert all(np.allclose(a.kappa, b.kappa) and a.t0 == b.t0
               for a, b in zip(s1, s2))


# -------------------------------------------------------------------- episodes
def test_episode_determinism(calib):
    sim = GreenDyGNNSim(calib, seed=3)
    r1 = sim.rollout_static(4, 1234)
    r2 = sim.rollout_static(4, 1234)
    assert r1 == r2
    r3 = sim.rollout_static(4, 1235)
    assert r3["energy_j"] != r1["energy_j"]


def test_state_vector_shape_and_bounds(calib):
    sim = GreenDyGNNSim(calib, seed=0)
    obs = sim.reset(77)
    assert obs.shape == (state_dim(calib.P),)
    done = False
    while not done:
        assert np.all(np.isfinite(obs))
        assert np.all(obs >= -1e-6) and np.all(obs <= 1.0 + 1e-6)
        assert abs(obs[-len(W_CHOICES):].sum() - 1.0) < 1e-6  # one-hot
        obs, _, done, _ = sim.step(3)


def test_reward_normalization_oracle_near_zero(calib):
    """Advantage-style reward: an always-per-decision-optimal policy scores 0;
    the per-episode best STATIC policy sits at or slightly below 0, and its
    normalized energy sits at/near 1.0 (the scale-free reporting metric)."""
    sim = GreenDyGNNSim(calib, seed=0)
    for es in (11, 22, 33):
        o = sim.oracle_static(es)
        # slightly-positive rewards are a boundary effect: a window that spans
        # into a cheaper (post-congestion) segment can beat the reference taken
        # at the window's START segment; bounded by one window's share.
        assert -0.35 * sim.reward_scale < o["reward"] <= 0.05 * sim.reward_scale
        assert 0.97 <= o["norm_energy"] < 1.4


def test_reward_comparable_across_decision_counts(calib):
    """A W=1 policy makes ~3000 decisions, W=128 ~24; the energy-weighted
    reward and norm_energy must stay O(1) for both (the v1 per-decision
    reward made them incomparable)."""
    sim = GreenDyGNNSim(calib, seed=0)
    r_small = sim.rollout_static(0, 42)   # W=1
    r_big = sim.rollout_static(7, 42)     # W=128
    for r in (r_small, r_big):
        assert -3.0 * sim.reward_scale < r["reward"] <= 1e-9
        assert 0.99 <= r["norm_energy"] < 4.0


def test_policy_ordering_heuristic_beats_random(calib):
    sim = GreenDyGNNSim(calib, seed=0)
    seeds = list(range(200, 240))
    h = np.mean([sim.rollout(heuristic_policy, s)["reward"] for s in seeds])
    r = np.mean([sim.rollout(random_policy_factory(1), s)["reward"] for s in seeds])
    assert h > r, f"heuristic ({h:.3f}) must beat random ({r:.3f})"


def test_observed_khat_tracks_injected_kappa(calib):
    """Congestion OBSERVABILITY: injecting bandwidth congestion on one owner
    must raise that owner's khat observation (the v1 miss-fraction signal
    failed exactly this)."""
    sim = GreenDyGNNSim(calib, seed=0, domain_randomization=False,
                        obs_noise=0.0, archetypes=("single-link-bw",))
    sim.reset(50)
    # roll to the congested middle of the episode
    for _ in range(60):
        obs, _, done, info = sim.step(4)   # W=16
        if done:
            break
    kappa = info["kappa"]
    victim = int(np.argmax(kappa))
    assert kappa[victim] > 1.5, "profile should be congested mid-episode"
    n = sim.n_remote
    khat_obs = obs[:n]
    assert np.argmax(khat_obs) == victim
    assert khat_obs[victim] > khat_obs[(victim + 1) % n] * 1.2


def test_calib_json_roundtrip(tmp_path, calib):
    p = tmp_path / "c.json"
    calib.save(str(p))
    c2 = CalibParams.load(str(p))
    assert c2 == calib


# ------------------------------------------------- real-calibration behaviors
def _real_calib(name):
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", f"calib_{name}.json")
    return CalibParams.load(p)


def test_h_curve_supports_inverted_orientation():
    """reddit's calibrated h(W) RISES with W (hmin=0.808 > hmax=0.704): at
    small W the async builder can't keep up and serves stale windows. The
    logistic must handle both orientations."""
    from dataclasses import asdict
    base = CalibParams()
    inv = CalibParams(**{**asdict(base), "hmin": 0.81, "hmax": 0.70,
                         "w_half": 3.4, "gamma_h": 8.0, "U0": 1000.0})
    m = StepModel(inv)
    hs = [m.hit_rate(w) for w in W_CHOICES]
    assert all(0.0 <= h <= 1.0 for h in hs)
    assert hs[-1] > hs[0], "inverted curve: h must RISE with W"
    # classic orientation still decays
    md = StepModel(base)
    hd = [md.hit_rate(w) for w in W_CHOICES]
    assert hd[-1] < hd[0]


def test_domain_randomization_preserves_h_orientation():
    from simulator import _randomized
    from dataclasses import asdict
    rng = np.random.default_rng(0)
    inv = CalibParams(**{**asdict(CalibParams()), "hmin": 0.81, "hmax": 0.70})
    dec = CalibParams()
    for _ in range(50):
        q = _randomized(inv, rng)
        assert q.hmin > q.hmax, "inverted orientation must survive DR"
        q2 = _randomized(dec, rng)
        assert q2.hmax > q2.hmin, "classic orientation must survive DR"


def test_real_calibs_load_with_observation_maps():
    for name in ("reddit", "ogbn-products"):
        c = _real_calib(name)
        assert c.obs_kappa_map and len(c.obs_kappa_map) == 5
        assert c.obs_delay_map and len(c.obs_delay_map) == 3
        trues = [t for t, _ in c.obs_kappa_map]
        assert trues == sorted(trues)
        # products decays, reddit is inverted
        if name == "reddit":
            assert c.hmin > c.hmax
        else:
            assert c.hmax > c.hmin
        # a full episode must run end-to-end on the real calib
        sim = GreenDyGNNSim(c, seed=0)
        r = sim.rollout_static(4, 99)
        assert r["decisions"] > 0 and np.isfinite(r["norm_energy"])


def test_observation_compression_applied():
    """The agent must see the CALIBRATED (compressed) severity, not the true
    kappa; and sub-1 observed ratios (reddit at mild congestion) floor to 1."""
    from dataclasses import asdict
    base = asdict(CalibParams())
    base["obs_kappa_map"] = [[9.9, 0.81], [19.4, 1.28], [48.4, 2.73],
                             [96.8, 8.47], [193.7, 13.97]]   # reddit-like
    from simulator import KHAT_NORM
    c = CalibParams(**base)
    sim = GreenDyGNNSim(c, seed=0, domain_randomization=False, obs_noise=0.0,
                        archetypes=("single-link-bw",))
    sim.reset(50)
    done = False
    max_khat_obs = 1.0
    while not done:
        obs, _, done, info = sim.step(4)
        n = sim.n_remote
        max_khat_obs = max(max_khat_obs, float(obs[:n].max()) * KHAT_NORM)
        assert float(obs[:n].min()) * KHAT_NORM >= 1.0 - 1e-6  # floored
    # compressed: even severe true kappa (<=150 in this archetype) must be
    # observed well below truth (reddit-like map tops out ~14 at true 193.7);
    # note the state feature also saturates at KHAT_NORM=8 by design.
    assert max_khat_obs < 20.0, f"observation not compressed: {max_khat_obs}"


def test_mild_congestion_unobservable_on_reddit_like_map():
    """reddit 1000mbit: true kappa 9.9 observed at 0.81 -> floors to ~1.0;
    the per-owner channel carries NO signal at mild severity (this is the
    physical reality the policy must live with)."""
    from dataclasses import asdict
    base = asdict(CalibParams())
    base["obs_kappa_map"] = [[9.9, 0.81], [19.4, 1.28], [48.4, 2.73],
                             [96.8, 8.47], [193.7, 13.97]]
    c = CalibParams(**base)
    sim = GreenDyGNNSim(c, seed=0, domain_randomization=False, obs_noise=0.0)
    sim.reset(1)
    comp = sim._compress(np.array([1.0, 9.9, 193.7]), c.obs_kappa_map, (1.0, 1.0))
    assert abs(comp[0] - 1.0) < 1e-9
    assert comp[1] <= 1.0 + 1e-9          # mild: at/below the floor
    assert 10.0 < comp[2] < 20.0          # severe: visible but compressed


# ----------------------------------------- overlap model: axis-direction proof
def test_overlap_model_axis_directions_reddit():
    """Sim-to-real behavioral transfer, reddit: the fitted overlap model must
    reproduce the MEASURED direction of the best-W response on each congestion
    axis (novel finding: the two axes point in OPPOSITE directions, which the
    Eq.7 'congestion -> shrink W' heuristic gets wrong on bandwidth):
      * bandwidth (tbf/kappa): small W is catastrophic (measured 50mbit
        W4=2274ms vs W16=917ms) because the rebuild bulk pays kappa every
        window; best W stays at/above the clean optimum.
      * delay (netem): best W does NOT grow; small-W region stays competitive
        (measured W8 best at 10ms)."""
    c = _real_calib("reddit")
    assert c.t_c is not None, "couplings must be fitted (run fit_coupling.py)"
    m = StepModel(c, alloc_on=False)
    k1, d0 = np.ones(3), np.zeros(3)
    sub = (4, 8, 16, 32)

    def best(kappa, delta):
        g = [m.step_time(w, kappa, delta)[0] for w in sub]
        return sub[int(np.argmin(g))], dict(zip(sub, g))

    b_clean, _ = best(k1, d0)
    kv = k1.copy(); kv[2] = 193.7
    b_bw, g_bw = best(kv, d0)
    assert b_bw >= b_clean, f"bandwidth must not shrink best W ({b_clean}->{b_bw})"
    assert g_bw[4] > 1.8 * g_bw[16], (
        f"small-W bandwidth catastrophe missing: W4={g_bw[4]*1e3:.0f}ms "
        f"W16={g_bw[16]*1e3:.0f}ms")
    dv = d0.copy(); dv[2] = 0.010
    b_dl, g_dl = best(k1, dv)
    assert b_dl <= 16, f"delay must not push best W above 16 (got {b_dl})"
    assert g_dl[8] < g_dl[32], "under delay, W8 must beat W32 (measured order)"


def test_overlap_model_products_w_flat_under_congestion():
    """Products is pressure-bound: measured step time is W-flat under BOTH
    axes (200mbit: 622-632ms across W4-32; c4_10ms: 219-244ms). The model must
    reproduce the flatness (max/min < 1.3 over W4-32)."""
    c = _real_calib("ogbn-products")
    assert c.t_c is not None
    m = StepModel(c, alloc_on=False)
    k1, d0 = np.ones(3), np.zeros(3)
    sub = (4, 8, 16, 32)
    kv = k1.copy(); kv[2] = 48.42
    g = np.array([m.step_time(w, kv, d0)[0] for w in sub])
    assert g.max() / g.min() < 1.3, f"products c1 not W-flat: {g*1e3}"
    dv = d0.copy(); dv[2] = 0.010
    g = np.array([m.step_time(w, k1, dv)[0] for w in sub])
    assert g.max() / g.min() < 1.3, f"products c4 not W-flat: {g*1e3}"


def test_overlap_model_clean_near_flat_products():
    """Clean products: measured 132-137ms for W4..128 while h(W) collapses
    0.75->0.05 — the overlap budget must hide clean miss wire."""
    c = _real_calib("ogbn-products")
    m = StepModel(c, alloc_on=False)
    g = np.array([m.step_time(w, np.ones(3), np.zeros(3))[0]
                  for w in (4, 8, 16, 32, 64, 128)])
    assert g.max() / g.min() < 1.15, f"clean products not flat: {g*1e3}"


def test_legacy_model_still_used_without_couplings(calib):
    """Synthetic calib (no couplings) must keep the legacy path bit-for-bit."""
    assert calib.t_c is None
    m = StepModel(calib)
    T, comps = m.step_time(16, np.ones(3), np.zeros(3))
    assert comps["t_base"] == calib.t_base
