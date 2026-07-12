#!/usr/bin/env python3
"""Phase-0 gate (proposal §8) -- decomposed AND swept over congestion timescale.

Why a curve, not a point.  A single `stretch_len` silently splits the fixed
static->DP gap between two mechanisms:

  temporal adaptation : changing the single window as congestion drifts across
                        epochs.  GreenDyGNN's DQN already does this (w/o-RL
                        ablation: 6.9-8.6%).  Captured, at its oracle best, by the
                        per-stretch-uniform baseline.
  within-stretch        : at FIXED congestion, regions of the epoch's trace want
  non-uniformity          different W (hot-set turnover).  Trace-blind controllers
                        cannot get this; only the DP does.

`stretch_len` (epochs per constant-congestion stretch) controls how that split
falls: short stretches favour temporal, long stretches starve it.  So we sweep
`stretch_len` and report the two gaps as CURVES.  The headline is the novel
(per-stretch->DP) gap being *flat and > 0 across timescales* -- that pre-empts
"you picked a favourable stretch_len".

Decoupling timescale from congestion content (critical).  A single fixed ordered
regime list R is defined once; `stretch_len` only sets how many epochs each
consecutive regime spans (stretch k -> R[k % len(R)]).  Regimes are NEVER redrawn
per stretch_len, so the curves vary only with timescale, not with which links are
congested.

Four baselines under piecewise-stationary congestion (A7):
  1 global-static W   2 per-stretch W (oracle GreenDyGNN ceiling)
  3 per-epoch W       4 DP (non-uniform within epoch)
gap 1->2 temporal ; gap 2->4 NOVEL (headline) ; gap 3->4 within-epoch (tightest).

n_hot is a real cache-capacity setting (RapidGNN: 100k), not a knob -- run at the
deployed value and report what the curve says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration import CostModel
from .interval_cost import precompute, IntervalCosts
from .trace import Trace
from . import dp_solver as DP


W_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_STRETCH_LENS = [2, 3, 5, 7, 10, 15]


def default_regime_list(num_partitions: int, local_rank: int,
                        base_ms: float = 10.0) -> List[Tuple[str, np.ndarray]]:
    """One canonical, deterministic regime sequence R (proposal §9 archetypes).

    R[0] is clean (so every timescale starts clean, like the warmup); the rest
    cycle 15-25 ms on 1-2 owners.  This list is FIXED -- stretch_len only changes
    how long each entry is held, never which entry it is.
    """
    remote = [p for p in range(num_partitions) if p != local_rank]
    R: List[Tuple[str, np.ndarray]] = [("clean", np.ones(num_partitions))]
    patterns = ([{0: 20}, {len(remote) - 1: 25}, {1 % len(remote): 15, 0: 20},
                 {len(remote) - 1: 15, 1 % len(remote): 20}, {0: 25},
                 {1 % len(remote): 20}] if remote else [{}])
    for i, pat in enumerate(patterns):
        sg = np.ones(num_partitions)
        for idx, ms in pat.items():
            sg[remote[idx]] = 1.0 + ms / base_ms
        R.append((f"r{i + 1}", sg))
    return R


def congested_sigma(num_partitions: int, local_rank: int,
                    base_ms: float = 10.0) -> np.ndarray:
    """One fixed congested regime (1-2 owners delayed) -- the 'on' level of the
    square wave used by the timescale sweep."""
    remote = [p for p in range(num_partitions) if p != local_rank]
    sg = np.ones(num_partitions)
    if remote:
        sg[remote[0]] = 1.0 + 20.0 / base_ms
        if len(remote) > 1:
            sg[remote[1]] = 1.0 + 15.0 / base_ms
    return sg


def congestion_schedule(n_epochs: int, num_partitions: int, local_rank: int,
                        stretch_len: int,
                        regime_list: Optional[List[Tuple[str, np.ndarray]]] = None,
                        time_varying: bool = True, duty: float = 0.5,
                        sigma_c: Optional[np.ndarray] = None
                        ) -> Tuple[List[np.ndarray], List[int]]:
    """Assign each epoch a (sigma, stretch_id) as a SQUARE WAVE of FIXED exposure.

    Key design (decouples timescale from congestion content): exactly
    round(duty*n_epochs) epochs are congested and the rest clean -- this multiset
    is held FIXED across all stretch_len.  `stretch_len` changes only how those
    epochs are CLUSTERED (run length of each on/off block).  So the per-epoch
    congestion exposure is identical at every timescale; only the temporal
    arrangement differs.  This is what makes the novel (per-stretch->DP) gain
    genuinely timescale-invariant -- prefix-cycling a multi-regime list instead
    silently changes the exposure and confounds the curve.

    Stretches are maximal constant-sigma runs (so each is whole-epoch and
    single-sigma by construction; see stretch_leakage).
    """
    P = num_partitions
    if not time_varying:
        return [np.ones(P) for _ in range(n_epochs)], [0] * n_epochs
    sig_c = congested_sigma(P, local_rank) if sigma_c is None else np.asarray(sigma_c)

    n_cong = int(round(duty * n_epochs))
    pools = {"C": n_epochs - n_cong, "G": n_cong}
    seq, turn = [], "C"
    while len(seq) < n_epochs:
        if pools[turn] == 0:
            turn = "G" if turn == "C" else "C"
        take = min(stretch_len, pools[turn])
        seq.extend([turn] * take)
        pools[turn] -= take
        turn = "G" if turn == "C" else "C"

    sigmas = [np.ones(P) if t == "C" else sig_c.copy() for t in seq]
    sids, cur, prev = [], -1, None
    for t in seq:                       # new stretch id at each on/off transition
        if t != prev:
            cur += 1
            prev = t
        sids.append(cur)
    return sigmas, sids


def stretch_leakage(sigmas: List[np.ndarray], sids: List[int]) -> int:
    """Leakage guard: count stretches that are NOT whole-epoch / constant-sigma.

    Returns the number of violations (0 == clean).  A non-zero count means the
    per-stretch baseline would average a single W across epochs at *different*
    congestion levels, silently inflating the novel gap.
    """
    flags = 0
    for sid in sorted(set(sids)):
        idx = [e for e in range(len(sids)) if sids[e] == sid]
        # contiguous epochs
        if idx != list(range(idx[0], idx[-1] + 1)):
            flags += 1
            continue
        # constant sigma within the stretch
        s0 = sigmas[idx[0]]
        if any(not np.allclose(sigmas[e], s0) for e in idx[1:]):
            flags += 1
    return flags


def exposure_summary(sigmas: List[np.ndarray]) -> Dict[str, int]:
    """The congestion EXPOSURE = multiset of per-epoch sigma vectors.  Held fixed
    across stretch_len by the square wave; logged so that on REAL traces (where it
    may not be a clean square wave) we can see exactly what exposure each verdict
    was read under.  Returns {sigma_repr: epoch_count}."""
    out: Dict[str, int] = {}
    for s in sigmas:
        key = "[" + ",".join(f"{x:.2f}" for x in np.asarray(s)) + "]"
        out[key] = out.get(key, 0) + 1
    return out


def precompute_ics(traces: List[Trace], model: CostModel, n_hot: int,
                   w_max: int, templates=None) -> List[IntervalCosts]:
    """sigma-independent interval tables, ONE per epoch -- reused across the whole
    stretch sweep (cost matrices under any sigma are then cheap)."""
    return [precompute(tr, model, n_hot, w_max=w_max, templates=templates)
            for tr in traces]


@dataclass
class DatasetGateResult:
    dataset: str
    stretch_len: int
    n_stretches: int
    leakage_flags: int
    global_static_kJ: float
    per_stretch_kJ: float
    per_epoch_kJ: float
    dp_kJ: float
    temporal_gain_pct: float       # 1 -> 2  ORACLE per-stretch upper bound (exposure-
                                   #         fixed); NOT GreenDyGNN's realized gain
                                   #         (it has reaction lag; that's a later measure)
    novel_gain_pct: float          # 2 -> 4  (HEADLINE: leak-guarded irreducible gain)
    within_epoch_gain_pct: float   # 3 -> 4  (tightest isolation, stretch-invariant)
    per_stretch_vs_per_epoch_pct: float  # 2 -> 3  divergence; ~0 under square wave
                                   #         (per-stretch == per-epoch when every
                                   #         stretch is single-sigma), > 0 on real
                                   #         traces with within-stretch trace variation
    across_stretch_het: float
    within_stretch_het: float
    clean_dp_gain_pct: float
    per_stretch_W: List[int] = field(default_factory=list)
    exposure: Dict[str, int] = field(default_factory=dict)


def _evaluate(ics: List[IntervalCosts], model: CostModel, P: int, local: int,
              time_varying: bool, w_max: int, stretch_len: int,
              regime_list=None):
    n_epochs = len(ics)
    grid = [W for W in W_GRID if W <= w_max]
    sigmas, sids = congestion_schedule(n_epochs, P, local, stretch_len,
                                       regime_list, time_varying)
    leakage = stretch_leakage(sigmas, sids)

    dp_total = 0.0
    per_epoch_uniform_total = 0.0
    per_w_global = {W: 0.0 for W in grid}
    per_w_stretch: Dict[int, Dict[int, float]] = {}
    ppw_by_stretch: Dict[int, List[np.ndarray]] = {}

    for e, ic in enumerate(ics):
        cost, bt = ic.cost_matrix(model, sigmas[e], variant="A")
        dp_total += DP.solve_optionA(cost, bt, w_max=w_max).cost
        _, per_w = DP.oracle_uniform(cost, grid, bt)
        ew = min(per_w, key=per_w.get)
        per_epoch_uniform_total += per_w[ew]
        for W, v in per_w.items():
            per_w_global[W] += v
            per_w_stretch.setdefault(sids[e], {W2: 0.0 for W2 in grid})
            per_w_stretch[sids[e]][W] += v
        ppw_by_stretch.setdefault(sids[e], []).append(DP.per_position_w(cost, w_max))

    global_static = min(per_w_global.values())
    per_stretch, per_stretch_W = 0.0, []
    for sid in sorted(per_w_stretch):
        pw = per_w_stretch[sid]
        bw = min(pw, key=pw.get)
        per_stretch += pw[bw]
        per_stretch_W.append(int(bw))

    across_het = (float(np.std(per_stretch_W) / max(1e-9, np.mean(per_stretch_W)))
                  if per_stretch_W else 0.0)
    within_vars = [float(np.var(np.concatenate([p.astype(np.float64) for p in lst])))
                   for lst in ppw_by_stretch.values()]
    within_het = float(np.mean(within_vars)) if within_vars else 0.0

    return {"global_static": global_static, "per_stretch": per_stretch,
            "per_epoch": per_epoch_uniform_total, "dp": dp_total,
            "across_het": across_het, "within_het": within_het,
            "per_stretch_W": per_stretch_W, "n_stretches": len(per_stretch_W),
            "leakage": leakage, "exposure": exposure_summary(sigmas)}


def run_dataset(name: str, traces: List[Trace], model: CostModel, n_hot: int,
                w_max: Optional[int] = None, templates=None,
                stretch_len: int = 4,
                ics: Optional[List[IntervalCosts]] = None) -> DatasetGateResult:
    w_max = int(w_max or model.w_max)
    P, local = traces[0].num_partitions, traces[0].local_rank
    if ics is None:
        ics = precompute_ics(traces, model, n_hot, w_max, templates)
    g = _evaluate(ics, model, P, local, True, w_max, stretch_len)
    c = _evaluate(ics, model, P, local, False, w_max, stretch_len)

    def pct(hi, lo):
        return 100.0 * (hi - lo) / hi if hi > 0 else 0.0

    e = model.energy
    return DatasetGateResult(
        dataset=name, stretch_len=stretch_len, n_stretches=g["n_stretches"],
        leakage_flags=g["leakage"],
        global_static_kJ=e(g["global_static"]) / 1000.0,
        per_stretch_kJ=e(g["per_stretch"]) / 1000.0,
        per_epoch_kJ=e(g["per_epoch"]) / 1000.0,
        dp_kJ=e(g["dp"]) / 1000.0,
        temporal_gain_pct=pct(g["global_static"], g["per_stretch"]),
        novel_gain_pct=pct(g["per_stretch"], g["dp"]),
        within_epoch_gain_pct=pct(g["per_epoch"], g["dp"]),
        per_stretch_vs_per_epoch_pct=pct(g["per_stretch"], g["per_epoch"]),
        across_stretch_het=g["across_het"], within_stretch_het=g["within_het"],
        clean_dp_gain_pct=pct(c["global_static"], c["dp"]),
        per_stretch_W=g["per_stretch_W"], exposure=g["exposure"])


def sweep_dataset(name: str, traces: List[Trace], model: CostModel, n_hot: int,
                  w_max: Optional[int] = None, templates=None,
                  stretch_lens: Optional[List[int]] = None
                  ) -> List[DatasetGateResult]:
    """Run the gate across a range of congestion timescales, reusing one
    precompute.  Returns one DatasetGateResult per stretch_len (the curve)."""
    w_max = int(w_max or model.w_max)
    stretch_lens = stretch_lens or DEFAULT_STRETCH_LENS
    ics = precompute_ics(traces, model, n_hot, w_max, templates)
    return [run_dataset(name, traces, model, n_hot, w_max, templates, sl, ics=ics)
            for sl in stretch_lens]


# --------------------------------------------------------- curve diagnostics
def curve_diagnostics(curve: List[DatasetGateResult]) -> dict:
    """Summarise one dataset's curve: is temporal monotone-decreasing, is novel
    flat, any leakage?  These are the dry-run validation criteria."""
    sl = [r.stretch_len for r in curve]
    temporal = [r.temporal_gain_pct for r in curve]
    novel = [r.novel_gain_pct for r in curve]
    winep = [r.within_epoch_gain_pct for r in curve]
    pse = [r.per_stretch_vs_per_epoch_pct for r in curve]   # per-stretch -> per-epoch
    leak = sum(r.leakage_flags for r in curve)
    # Under the square wave every stretch is single-sigma, so per-stretch == per-epoch
    # (the four baselines collapse to three).  This divergence reopens on real traces
    # when epochs WITHIN a constant-sigma stretch want different W (trace variation):
    # then the leak-guarded novel gap (per-stretch->DP) is the strict number to report.
    collapsed = max(pse) < 0.5 if pse else True

    # With FIXED congestion exposure (square wave) the oracle per-stretch
    # advantage is exposure-determined, so temporal is ~timescale-invariant; we
    # only require it not to *rise* with timescale (tolerant to 2nd-decimal noise).
    tmean = float(np.mean(temporal))
    ttol = max(0.5, 0.1 * tmean)
    temporal_not_increasing = temporal[-1] <= temporal[0] + ttol
    # novel: flat == small spread relative to its level (no strong trend).  A
    # strong RISE with timescale would be the leakage/averaging signature.
    nmean = float(np.mean(novel))
    nspread = float(max(novel) - min(novel))
    novel_flat = nspread <= max(1.0, 0.5 * nmean)        # within 1pp or half the mean
    novel_rising = novel[-1] - novel[0]
    return {
        "stretch_lens": sl, "temporal_oracle_ub": [round(x, 2) for x in temporal],
        "novel": [round(x, 2) for x in novel], "within_epoch": [round(x, 2) for x in winep],
        "per_stretch_vs_per_epoch": [round(x, 2) for x in pse],
        "baselines_collapsed_per_stretch_eq_per_epoch": bool(collapsed),
        "leakage_total": leak, "temporal_not_increasing": bool(temporal_not_increasing),
        "temporal_mean": round(tmean, 2),
        "novel_flat": bool(novel_flat), "novel_mean": round(nmean, 2),
        "novel_spread": round(nspread, 2), "novel_rise_short_to_long": round(novel_rising, 2),
        "exposure": curve[0].exposure if curve else {},
    }


def floor_report(traces: List[Trace], model: CostModel, n_hot: int,
                 w_max: Optional[int] = None) -> dict:
    """Communication-floor view logged BESIDE the novel-gain curves (Arch A/F/§15).

    For each epoch trace: the exact floor L_0 (Thm A, schedule-free) and, for the
    DP schedule under clean and congested sigma, the near-floor decomposition
    (rho, gamma_R, phi, eta; Thm F) plus working-set drift beta.  Averaged over
    epochs.  rho>=1 is the sandwich; eta<1 flags the initiation-dominated regime.
    """
    from .floor import (communication_floor, schedule_transfers, near_floor,
                        working_set_drift)
    from .interval_cost import precompute
    w_max = int(w_max or model.w_max)
    P, local = traces[0].num_partitions, traces[0].local_rank
    conds = {"clean": np.ones(P), "cong": congested_sigma(P, local)}
    acc = {k: {"L0": [], "rho": [], "gamma_R": [], "phi": [], "eta": []} for k in conds}
    betas = []
    for tr in traces:
        fl = communication_floor(tr, model)
        ic = precompute(tr, model, n_hot, w_max=w_max)   # sigma-independent, reused
        for k, sg in conds.items():
            cost, bt = ic.cost_matrix(model, sg, variant="A")
            sched = DP.solve_optionA(cost, bt, w_max=w_max)
            tf = schedule_transfers(tr, sched.windows, model, n_hot, sigma=sg,
                                    rebuild="delta")
            nf = near_floor(fl, tf, model)
            acc[k]["L0"].append(fl.L0); acc[k]["rho"].append(nf.rho)
            acc[k]["gamma_R"].append(nf.gamma_R); acc[k]["phi"].append(nf.phi)
            acc[k]["eta"].append(nf.eta)
        betas.append((working_set_drift(tr, 8), working_set_drift(tr, 32)))
    out = {}
    for k in conds:
        out[k] = {m: round(float(np.mean(acc[k][m])), 4) for m in acc[k]}
        out[k]["initiation_dominated"] = bool(np.mean(acc[k]["eta"]) < 1.0)
    out["beta8"] = round(float(np.mean([b[0] for b in betas])), 3)
    out["beta32"] = round(float(np.mean([b[1] for b in betas])), 3)
    out["n_star_mean"] = round(float(np.mean(model.n_star())), 1)
    return out


def decision_from_curves(diags: Dict[str, dict]) -> dict:
    """PRE-REGISTERED verdict rule (see GATE_PREREGISTRATION.md, Rule 1).

    A dataset CLEARS iff novel_mean >= 3% AND novel_flat AND leakage_total == 0.
    Count clearing datasets:  >=2 -> STRONG ; ==1 -> PER-DATASET ; 0 -> FALLBACK.
    This is committed before the data exists; the code emits the verdict so it is
    not chosen post-hoc.
    """
    per_ds = {}
    for ds, d in diags.items():
        clears = (d["novel_flat"] and d["novel_mean"] >= 3.0
                  and d["leakage_total"] == 0)
        per_ds[ds] = "CLEARS" if clears else "no"
    n_clear = sum(v == "CLEARS" for v in per_ds.values())
    if n_clear >= 2:
        verdict = ("STRONG — within-epoch non-uniformity clears on >=2/3 datasets; "
                   "universal exact-optimal non-uniform-scheduling headline holds")
    elif n_clear == 1:
        verdict = ("PER-DATASET — novel gain is real but graph-dependent (clears on 1/3); "
                   "characterise WHICH graphs (within-epoch locality turnover), not universal")
    else:
        verdict = ("FALLBACK — novel gain <3% everywhere; lead with the no-regret GUARANTEE "
                   "replacing the DQN + OPTIMAL per-owner allocation")
    return {"verdict": verdict, "n_clearing": n_clear, "per_dataset": per_ds,
            "rule": "clears iff novel>=3% & flat & no-leak; >=2 STRONG / ==1 PER-DATASET / 0 FALLBACK"}


def _spearman(x: List[float], y: List[float]) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2:
        return 0.0
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def predictor_pooled(curves_by: Dict[str, List[DatasetGateResult]]) -> dict:
    """Rule 2: the predictor claim rests on the POOLED relationship across all
    (dataset x stretch_len) points -- within-stretch W*(t) variance vs novel gap --
    not the 3-point cross-dataset ordering.  Reports Spearman over the pooled set
    and, separately, the (weaker) 3-point per-dataset-mean ordering for contrast.
    """
    het, nov = [], []
    for curve in curves_by.values():
        for r in curve:
            het.append(r.within_stretch_het)
            nov.append(r.novel_gain_pct)
    pooled = _spearman(het, nov)
    ds_het = [float(np.mean([r.within_stretch_het for r in c])) for c in curves_by.values()]
    ds_nov = [float(np.mean([r.novel_gain_pct for r in c])) for c in curves_by.values()]
    return {"pooled_spearman_within_het_vs_novel": round(pooled, 3),
            "n_points": len(het),
            "cross_dataset_spearman_3pt": round(_spearman(ds_het, ds_nov), 3),
            "note": "headline predictor claim rests on pooled n_points, NOT the 3pt ordering"}
