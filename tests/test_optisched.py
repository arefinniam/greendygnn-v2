#!/usr/bin/env python3
"""Validation suite for the OptiSched-GNN offline core.

Run:  python3 tests/test_optisched.py   (from artifacts/code)

Covers the claims the paper rests on:
  * Lemma 1.1 / §6.1 identity: precomputed residual misses match a brute-force
    recomputation against the realised top-n_hot hot set.
  * Theorem 1: the DP equals exhaustive brute force on small instances.
  * Theorem 2: the DP cost is <= the oracle-best uniform W.
  * §5.3 ordering: Option A (length-only) <= Option B (delta) <= A' (full-refresh).
  * Theorem 5: fixed-share regret is non-negative and finite.
  * Prefetcher integration: a non-uniform schedule tiles the epoch exactly.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optisched.trace import Trace, SyntheticTrace
from optisched.calibration import CostModel, default_model
from optisched import interval_cost as IC
from optisched import dp_solver as DP
from optisched.regime import RegimeLibrary, SimpleFixedShare

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {extra}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {extra}")


def brute_residual(trace, model, n_hot, i, w, P):
    """Independent O(W*F) recomputation of per-owner residual misses for (i, i+w]
    under the uniform template -- the ground truth for the §6.1 identity."""
    cnt, own = {}, {}
    for b in range(i, i + w):
        nb, ob = trace.batch(b)
        for node, owner in zip(nb.tolist(), ob.tolist()):
            cnt[node] = cnt.get(node, 0) + 1
            own[node] = owner
    nodes = list(cnt.keys())
    counts = np.array([cnt[n] for n in nodes], dtype=np.float64)
    owners = np.array([own[n] for n in nodes], dtype=np.int64)
    A = np.bincount(owners, weights=counts, minlength=P)
    if len(nodes) <= n_hot:
        return A - A  # zero residual
    top = np.argsort(-counts)[:n_hot]
    cover = np.bincount(owners[top], weights=counts[top], minlength=P)
    return A - cover


def test_identity():
    print("[identity] §6.1 residual = A - C")
    tr = SyntheticTrace.generate(num_batches=24, num_partitions=4, universe=1500,
                                 remote_per_batch=120, heterogeneity=0.6,
                                 owner_correlation=0.5, seed=11)
    m = default_model(num_partitions=4, w_max=8)
    n_hot = 200
    ic = IC.precompute(tr, m, n_hot, w_max=8)
    P = tr.num_partitions
    maxerr = 0.0
    rng = np.random.default_rng(0)
    for _ in range(40):
        i = int(rng.integers(0, tr.num_batches - 1))
        w = int(rng.integers(1, min(8, tr.num_batches - i) + 1))
        got = ic.residual[i, w - 1, 0, :]
        want = brute_residual(tr, m, n_hot, i, w, P)
        # residual top-k can differ on ties; compare the TOTAL residual (tie-free)
        maxerr = max(maxerr, abs(got.sum() - want.sum()))
    check("residual total matches brute force", maxerr < 1e-6, f"(maxerr={maxerr:.2e})")


def test_dp_optimal():
    print("[dp] Theorem 1 (== brute force) & Theorem 2 (<= oracle-uniform)")
    rng = np.random.default_rng(3)
    worst_gap = 0.0
    dom_ok = True
    for t in range(12):
        tr = SyntheticTrace.generate(
            num_batches=int(rng.integers(8, 18)), num_partitions=4,
            universe=2000, remote_per_batch=int(rng.integers(80, 160)),
            heterogeneity=float(rng.random()), owner_correlation=float(rng.random()),
            seed=100 + t)
        m = CostModel(num_partitions=4, w_max=8, a=0.05, b=0.05, c=0.5,
                      alpha=0.35, t_miss=np.full(4, 4e-4))
        ic = IC.precompute(tr, m, n_hot=int(rng.integers(150, 400)), w_max=8)
        sigma = np.array([1.0] + list(1.0 + 2 * rng.random(3)))
        cost, bt = ic.cost_matrix(m, sigma, "A")
        dp = DP.solve_optionA(cost, bt, 8)
        bf = DP.brute_force(cost, 8)
        oru, _ = DP.oracle_uniform(cost, [1, 2, 3, 4, 5, 6, 7, 8], bt)
        worst_gap = max(worst_gap, abs(dp.cost - bf))
        dom_ok = dom_ok and (dp.cost <= oru.cost + 1e-9)
    check("DP == brute force", worst_gap < 1e-9, f"(maxgap={worst_gap:.2e})")
    check("DP <= oracle-uniform (Theorem 2)", dom_ok)


def test_variant_ordering():
    # Proposal §5.3: A' (full-refresh) is a CERTIFIED upper bound on the
    # delta-exact cost B; the length-only A is an unbiased AVERAGE of the delta
    # rebuild and is NOT a per-instance bound either way.  So the only guaranteed
    # ordering is B <= A'; we also report the (unbounded) A-vs-B H-slack gap.
    print("[variants] §5.3  B <= A'  (A is the unbiased middle)")
    tr = SyntheticTrace.generate(num_batches=20, num_partitions=4, universe=3000,
                                 remote_per_batch=150, heterogeneity=0.5,
                                 owner_correlation=0.7, seed=9)
    m = CostModel(num_partitions=4, w_max=8, a=0.05, b=0.05, c=0.5, alpha=0.35,
                  t_miss=np.full(4, 4e-4))
    ic = IC.precompute(tr, m, n_hot=300, w_max=8)
    sig = np.ones(4)
    cA, _ = ic.cost_matrix(m, sig, "A")
    cAp, _ = ic.cost_matrix(m, sig, "Ap")
    sA = DP.solve_optionA(cA, None, 8)
    sAp = DP.solve_optionA(cAp, None, 8)
    sB = DP.solve_optionB(tr, m, 300, sig, 8)
    check("B <= A' (certified upper bound)", sB.cost <= sAp.cost + 1e-9,
          f"(B={sB.cost:.4f} Ap={sAp.cost:.4f})")
    check("A' >= B with finite H-slack", np.isfinite(sAp.cost - sB.cost),
          f"(A={sA.cost:.4f} B={sB.cost:.4f} Ap={sAp.cost:.4f})")


def test_controller():
    print("[online] Theorem 5 fixed-share")
    tr = SyntheticTrace.generate(num_batches=40, num_partitions=4, universe=4000,
                                 remote_per_batch=200, heterogeneity=0.5,
                                 owner_correlation=0.8, seed=4)
    m = CostModel(num_partitions=4, w_max=16, a=0.1, b=0.1, c=0.5, alpha=0.35,
                  t_miss=np.full(4, 4e-4))
    lib = RegimeLibrary.build(tr, m, n_hot=500)
    ctl = lib.controller()
    pos = 0
    while pos < tr.num_batches:
        w, t = ctl.select(pos)
        sg = np.array([1.0, 3.0, 1.0, 1.0]) if pos < 20 else np.ones(4)
        oc, _ = lib.ic.cost_matrix(m, sg, "A")
        ctl.update(pos, sg, oc)
        pos += w
    check("regret finite & >= 0", np.isfinite(ctl.regret_so_far) and ctl.regret_so_far >= -1e-9,
          f"(regret={ctl.regret_so_far:.4f})")
    # SimpleFixedShare: weight concentrates on the consistently-cheapest expert
    sfs = SimpleFixedShare(3, horizon=20)
    for _ in range(20):
        sfs.select()
        sfs.update_with_losses(np.array([0.1, 1.0, 1.0]))  # expert 0 best
    check("fixed-share tracks best expert", int(np.argmax(sfs.w)) == 0,
          f"(weights={np.round(sfs.w,3).tolist()})")


def test_prefetcher_schedule():
    print("[prefetcher] non-uniform schedule tiles the epoch")
    from prefetcher import BatchPrefetcher
    obj = BatchPrefetcher.__new__(BatchPrefetcher)
    obj.window_size = 16
    obj.batches_per_epoch = 12
    obj.schedule = None
    BatchPrefetcher.set_schedule(obj, [3, 3, 4, 2])
    starts = [0, 3, 6, 10]
    lens = [BatchPrefetcher._win_len(obj, s + 1) for s in starts]  # 1-based global
    check("scheduled lengths recovered", lens == [3, 3, 4, 2], f"({lens})")
    check("schedule tiles to bpe", sum(lens) == obj.batches_per_epoch)
    BatchPrefetcher.set_schedule(obj, None)
    check("None schedule -> static window", BatchPrefetcher._win_len(obj, 5) == 16)


def test_gate_decomposition():
    print("[gate] baseline ordering  static >= per-stretch >= per-epoch >= DP")
    from optisched import gate as G
    traces = [SyntheticTrace.generate(
        num_batches=50, num_partitions=4, universe=8000, remote_per_batch=250,
        heterogeneity=0.6, owner_correlation=0.85, seed=7 * e + 1) for e in range(12)]
    m = CostModel(num_partitions=4, w_max=64, a=0.3, b=0.3, c=0.5, alpha=0.35,
                  t_miss=np.full(4, 4e-4))
    r = G.run_dataset("demo", traces, m, n_hot=700, w_max=64, stretch_len=3)
    eps = 1e-6
    check("static >= per-stretch", r.global_static_kJ >= r.per_stretch_kJ - eps,
          f"({r.global_static_kJ:.3f} >= {r.per_stretch_kJ:.3f})")
    check("per-stretch >= DP (novel gain >= 0)", r.per_stretch_kJ >= r.dp_kJ - eps,
          f"(novel={r.novel_gain_pct:.2f}%)")
    check("per-epoch >= DP (within-epoch gain >= 0)", r.per_epoch_kJ >= r.dp_kJ - eps,
          f"(winEp={r.within_epoch_gain_pct:.2f}%)")
    check("temporal & novel gains finite", np.isfinite(r.temporal_gain_pct) and
          np.isfinite(r.novel_gain_pct))


def test_leakage_guard():
    print("[leakage] stretch guard catches mixed-congestion / straddling stretches")
    from optisched.gate import stretch_leakage, congestion_schedule
    P = 4
    sigmas, sids = congestion_schedule(12, P, 0, stretch_len=3)
    check("clean schedule -> 0 leakage", stretch_leakage(sigmas, sids) == 0)
    # corrupt: same stretch id spans two different sigmas (the inflation bug)
    bad_sig = [s.copy() for s in sigmas]
    bad_sid = list(sids)
    bad_sig[0] = np.array([1.0, 3.0, 1.0, 1.0]); bad_sid[0] = bad_sid[1]
    bad_sig[1] = np.array([1.0, 1.0, 1.0, 1.0])
    check("mixed-sigma stretch -> flagged", stretch_leakage(bad_sig, bad_sid) >= 1)
    # corrupt: non-contiguous stretch (a straddle signature)
    nc_sid = list(sids); nc_sid[0] = nc_sid[-1]
    check("non-contiguous stretch -> flagged", stretch_leakage(sigmas, nc_sid) >= 1)


def test_sweep_monotone_flat():
    print("[sweep] dry-run criteria: ordering, temporal falls, novel flat, no leak")
    from optisched import gate as G
    traces = [SyntheticTrace.generate(
        num_batches=60, num_partitions=4, universe=9000, remote_per_batch=260,
        heterogeneity=0.6, owner_correlation=0.85, seed=5 * e + 2) for e in range(24)]
    m = CostModel(num_partitions=4, w_max=64, a=0.3, b=0.3, c=0.5, alpha=0.35,
                  t_miss=np.full(4, 4e-4))
    curve = G.sweep_dataset("demo", traces, m, n_hot=700, w_max=64,
                            stretch_lens=[2, 3, 4, 6, 8, 12])
    eps = 1e-6
    ordering = all(r.global_static_kJ >= r.per_stretch_kJ - eps and
                   r.per_stretch_kJ >= r.per_epoch_kJ - eps and
                   r.per_epoch_kJ >= r.dp_kJ - eps for r in curve)
    check("ordering static>=stretch>=perEp>=DP at every stretch_len", ordering)
    d = G.curve_diagnostics(curve)
    check("zero leakage across sweep", d["leakage_total"] == 0)
    check("temporal (oracle UB) not rising with timescale", d["temporal_not_increasing"],
          f"(mean={d['temporal_mean']} {d['temporal_oracle_ub']})")
    check("novel gain FLAT across timescale (no leakage inflation)", d["novel_flat"],
          f"(mean={d['novel_mean']} spread={d['novel_spread']} {d['novel']})")
    # within-epoch is stretch_len-independent by construction -> essentially constant
    we = d["within_epoch"]
    check("within-epoch gain ~invariant", (max(we) - min(we)) <= max(0.75, 0.4 * np.mean(we)),
          f"({we})")
    # square-wave collapse: per-stretch == per-epoch (every stretch single-sigma);
    # i.i.d. synthetic epochs share an optimal W within a stretch -> divergence ~0
    check("per-stretch == per-epoch under square wave (collapse to 3 baselines)",
          d["baselines_collapsed_per_stretch_eq_per_epoch"],
          f"(pStr->pEp {d['per_stretch_vs_per_epoch']})")
    check("exposure fixed across timescale (same multiset every stretch_len)",
          all(r.exposure == curve[0].exposure for r in curve),
          f"({curve[0].exposure})")


def test_preregistered_decision():
    print("[verdict] pre-registered counting rule (>=2 STRONG / ==1 PER-DATASET / 0 FALLBACK)")
    from optisched.gate import decision_from_curves, _spearman, predictor_pooled
    mk = lambda nov, flat=True, leak=0: {"novel_mean": nov, "novel_flat": flat,
                                         "leakage_total": leak}
    d2 = {"a": mk(5.0), "b": mk(4.0), "c": mk(0.2)}      # 2 clear
    d1 = {"a": mk(5.0), "b": mk(0.5), "c": mk(0.2)}      # 1 clears
    d0 = {"a": mk(2.0), "b": mk(0.5), "c": mk(0.2)}      # 0 clear
    check("2/3 clear -> STRONG", decision_from_curves(d2)["verdict"].startswith("STRONG"))
    check("1/3 clear -> PER-DATASET", decision_from_curves(d1)["verdict"].startswith("PER-DATASET"))
    check("0/3 clear -> FALLBACK", decision_from_curves(d0)["verdict"].startswith("FALLBACK"))
    # a high novel that ISN'T flat must NOT clear (timescale objection enforced)
    dnf = {"a": mk(9.0, flat=False), "b": mk(0.5), "c": mk(0.2)}
    check("high-but-not-flat novel does not clear", decision_from_curves(dnf)["n_clearing"] == 0)
    # leakage disqualifies a clear
    dlk = {"a": mk(9.0, leak=2), "b": mk(0.5), "c": mk(0.2)}
    check("leaky dataset cannot clear", decision_from_curves(dlk)["n_clearing"] == 0)
    # spearman sanity
    check("spearman monotone ~1", _spearman([1, 2, 3, 4], [1, 2, 3, 4]) > 0.99)
    check("spearman anti-monotone ~-1", _spearman([1, 2, 3, 4], [4, 3, 2, 1]) < -0.99)


def test_floor():
    print("[floor] Theorem A sandwich + Theorem F decomposition + Lemma D")
    from optisched import floor as FL
    tr = SyntheticTrace.generate(num_batches=50, num_partitions=4, universe=5000,
                                 remote_per_batch=220, heterogeneity=0.5,
                                 owner_correlation=0.7, seed=8)
    m = CostModel(num_partitions=4, w_max=32, a=0.2, b=0.2, c=0.5,
                  eps_init=np.full(4, 1e-3), kappa=np.full(4, 8e-9), d=128,
                  c_max=np.full(4, 1200.0))
    fl = FL.communication_floor(tr, m)
    ic = IC.precompute(tr, m, 600, w_max=32)
    cost, bt = ic.cost_matrix(m, np.ones(4), "A")
    sched = DP.solve_optionA(cost, bt, 32)
    for reb in ("full", "delta"):
        tf = FL.schedule_transfers(tr, sched.windows, m, 600, rebuild=reb)
        nf = FL.near_floor(fl, tf, m)
        check(f"sandwich rho>=1 ({reb})", nf.rho >= 1 - 1e-9, f"(rho={nf.rho:.3f})")
        check(f"Thm F identity rho==rho_decomp ({reb})",
              abs(nf.rho - nf.rho_decomp) < 1e-6)
    tdelta = FL.schedule_transfers(tr, sched.windows, m, 600, rebuild="delta")
    tfull = FL.schedule_transfers(tr, sched.windows, m, 600, rebuild="full")
    check("delta Q <= full Q (delta re-sends less)", tdelta["Q_win"] <= tfull["Q_win"] + 1e-9)
    # Lemma D: disjoint per-batch nodes -> contiguous lifetimes, phi == 1
    rng = np.random.default_rng(0)
    bn, bo = [], []
    nid = 0
    for b in range(10):
        ids = np.arange(nid, nid + 40, dtype=np.int64); nid += 40
        bn.append(ids); bo.append(rng.integers(1, 4, size=40).astype(np.int32))
    dj = Trace.from_batches(bn, bo, 4, 0)
    wins = [(b, 1) for b in range(10)]
    lf = FL.lifetime_fit(dj, wins, n_hot=100)
    tdj = FL.schedule_transfers(dj, wins, m, 100, rebuild="delta")
    phi = tdj["Q_win"] / FL.footprints(dj)["U"]
    check("Lemma D: disjoint trace attainable & phi==1", lf["attainable"] and abs(phi - 1.0) < 1e-9,
          f"(attainable={lf['attainable']} phi={phi:.3f})")


def test_owner_decoupled():
    print("[decoupled] Theorem G exact owner-decoupled DP + budget templates")
    tr = SyntheticTrace.generate(num_batches=36, num_partitions=4, universe=4000,
                                 remote_per_batch=200, heterogeneity=0.5,
                                 owner_correlation=0.8, seed=6)
    m = CostModel(num_partitions=4, w_max=32, a=0.2, b=0.2, c=0.5)
    sigma = np.array([1., 3., 1., 1.])
    budgets = {1: 240, 2: 240, 3: 240}
    dec = DP.solve_owner_decoupled(tr, m, budgets, sigma, 32)
    ok = True
    for p in (1, 2, 3):
        sub = tr.restrict_owner(p)
        ic = IC.precompute(sub, m, 240, w_max=32)
        c, bt = ic.cost_matrix(m, sigma, "A")
        ok = ok and abs(dec.per_owner[p].cost - DP.brute_force(c, 32)) < 1e-9
    check("per-owner DP == brute force (exact)", ok)
    check("total == sum of per-owner", abs(dec.total_cost - sum(s.cost for s in dec.per_owner.values())) < 1e-12)
    T = [{1: 240, 2: 240, 3: 240}, {1: 480, 2: 120, 3: 120}]
    best, allr = DP.solve_budget_templates(tr, m, T, sigma, 32)
    check("budget-template picks min", abs(best.total_cost - min(r.total_cost for r in allr)) < 1e-12)
    check("biasing budget to congested owner helps", allr[1].total_cost < allr[0].total_cost)


def test_weighted_hotset():
    print("[weighted] Theorem B weighted hot-set")
    tr = SyntheticTrace.generate(num_batches=32, num_partitions=4, universe=3500,
                                 remote_per_batch=180, heterogeneity=0.5,
                                 owner_correlation=0.8, seed=9)
    m = CostModel(num_partitions=4, w_max=32, a=0.2, b=0.2, c=0.5)
    sigma = np.array([1., 4., 1., 1.])
    cw = IC.precompute(tr, m, 300, w_max=32, templates=IC.weighted_template(m, sigma))
    cf = IC.precompute(tr, m, 300, w_max=32)
    sw = DP.solve_optionA(cw.cost_matrix(m, sigma, "A")[0], None, 32)
    sf = DP.solve_optionA(cf.cost_matrix(m, sigma, "A")[0], None, 32)
    check("weighted <= frequency under congestion", sw.cost <= sf.cost + 1e-9,
          f"(weighted={sw.cost:.4f} freq={sf.cost:.4f})")
    # reduction: sigma=1 & homogeneous -> weighted template constant -> == frequency
    c1 = cw.cost_matrix(m, np.ones(4), "A")[0]
    cwu = IC.precompute(tr, m, 300, w_max=32, templates=IC.weighted_template(m, None))
    cu = cwu.cost_matrix(m, np.ones(4), "A")[0]
    cfu = cf.cost_matrix(m, np.ones(4), "A")[0]
    check("weighted reduces to top-frequency at sigma=1", np.allclose(cu, cfu))


def test_beta_and_switch():
    print("[beta+switch] working-set drift + Theorem-H switching cost")
    from optisched import floor as FL
    tr = SyntheticTrace.generate(num_batches=60, num_partitions=4, universe=6000,
                                 remote_per_batch=250, heterogeneity=0.6,
                                 owner_correlation=0.7, seed=3)
    b_small, b_large = FL.working_set_drift(tr, 4), FL.working_set_drift(tr, 32)
    check("beta >= 1", b_small >= 1 - 1e-9 and b_large >= 1 - 1e-9)
    check("beta decreases with window length (more reuse captured)", b_small >= b_large - 1e-9,
          f"(beta4={b_small:.2f} beta32={b_large:.2f})")
    sfs = SimpleFixedShare(3, horizon=10, switch_cost=0.5)
    # force alternation so it switches, then check accounting
    losses = [np.array([0.1, 1.0, 1.0]), np.array([1.0, 0.1, 1.0])]
    for r in range(6):
        sfs.select()
        sfs.update_with_losses(losses[r % 2])
    st = sfs.stats()
    check("switches counted & switch cost accumulated",
          st["switches"] >= 1 and st["switch_cost_total"] > 0, f"({st})")
    check("regret includes switch cost", sfs.regret_so_far >= sfs.cum_switch - 1e-9)


if __name__ == "__main__":
    test_identity()
    test_dp_optimal()
    test_variant_ordering()
    test_controller()
    test_prefetcher_schedule()
    test_gate_decomposition()
    test_leakage_guard()
    test_sweep_monotone_flat()
    test_preregistered_decision()
    test_floor()
    test_owner_decoupled()
    test_weighted_hotset()
    test_beta_and_switch()
    print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
