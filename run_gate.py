#!/usr/bin/env python3
"""OptiSched-GNN Phase-0 GATE driver (proposal §8, §13 Phase 0).

Run FIRST. Measures the clairvoyant ceiling -- the DP-optimal non-uniform
schedule's energy gain over the oracle-best uniform W -- under clean and
time-varying congestion, plots gain vs temporal heterogeneity, and applies the
committed decision rule.

Two trace sources:

  Synthetic (no cluster):
      python3 run_gate.py --synthetic --epochs 10

  Real traces dumped by dump_trace.py (per dataset, per epoch .npz):
      traces/<dataset>/epoch_000.npz, epoch_001.npz, ...
      python3 run_gate.py --traces traces --model calib.json \
          --datasets reddit,ogbn-products,ogbn-papers100M --n_hot 100000

Outputs: a printed gain table + decision, a JSON results file, and a PDF figure.
"""

import argparse
import glob
import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np

from optisched.calibration import CostModel, default_model
from optisched.trace import Trace, SyntheticTrace
from optisched.interval_cost import make_templates
from optisched import gate as G

_HERE = os.path.dirname(os.path.abspath(__file__))


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "MISSING"


def provenance():
    """Bind each result to the exact rules + code it was read under (so a real-
    trace verdict can never be silently re-read under different rules later)."""
    code = [os.path.join(_HERE, "optisched", f) for f in sorted(
        os.listdir(os.path.join(_HERE, "optisched"))) if f.endswith(".py")]
    code += [os.path.join(_HERE, "run_gate.py")]
    h = hashlib.sha256()
    for f in sorted(code):
        h.update(_sha256(f).encode())
    return {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preregistration_sha256": _sha256(os.path.join(_HERE, "GATE_PREREGISTRATION.md")),
        "ledger_sha256": _sha256(os.path.join(_HERE, "OPTISCHED_LEDGER.md")),
        "code_combined_sha256": h.hexdigest(),
    }


def load_real_traces(root: str, datasets: List[str], part: int = 0
                     ) -> Dict[str, List[Trace]]:
    out: Dict[str, List[Trace]] = {}
    for ds in datasets:
        # dump_trace.py / gen_real_trace.py write rank-prefixed r<part>_epoch_*.npz;
        # fall back to un-prefixed epoch_*.npz for hand-made traces.
        files = sorted(glob.glob(os.path.join(root, ds, f"r{part}_epoch_*.npz")))
        if not files:
            files = sorted(glob.glob(os.path.join(root, ds, "epoch_*.npz")))
        if not files:
            print(f"  [warn] no traces for {ds} under {root}/{ds}")
            continue
        out[ds] = [Trace.load(f) for f in files]
        print(f"  loaded {len(files)} epoch traces for {ds} "
              f"(B={out[ds][0].num_batches}, P={out[ds][0].num_partitions})")
    return out


def make_synthetic(epochs: int, num_partitions: int) -> Dict[str, List[Trace]]:
    """Three demo datasets spanning low->high temporal heterogeneity, so the
    gain-vs-heterogeneity relationship is visible without the cluster."""
    specs = {
        "reddit":          dict(universe=8000,  remote_per_batch=300, heterogeneity=0.15, corr=0.7, B=80),
        "ogbn-products":   dict(universe=20000, remote_per_batch=350, heterogeneity=0.5,  corr=0.8, B=120),
        "ogbn-papers100M": dict(universe=40000, remote_per_batch=400, heterogeneity=0.85, corr=0.85, B=160),
    }
    out: Dict[str, List[Trace]] = {}
    for ds, sp in specs.items():
        traces = []
        for e in range(epochs):
            traces.append(SyntheticTrace.generate(
                num_batches=sp["B"], num_partitions=num_partitions,
                universe=sp["universe"], remote_per_batch=sp["remote_per_batch"],
                heterogeneity=sp["heterogeneity"], owner_correlation=sp["corr"],
                seed=1000 * e + hash(ds) % 997))
        out[ds] = traces
    return out


def make_figure(curves_by, path):
    """curves_by: {dataset: [DatasetGateResult per stretch_len]}.  Two panels:
    temporal gain vs timescale (should fall) and novel gain vs timescale (should
    stay flat & >0 -- the timescale-invariant headline)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable, skipping figure: {e}")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    for ds, curve in curves_by.items():
        sl = [r.stretch_len for r in curve]
        ax1.plot(sl, [r.temporal_gain_pct for r in curve], "o-", label=ds)
        ax2.plot(sl, [r.novel_gain_pct for r in curve], "o-", label=ds)
    ax1.set_title("(a) temporal: ORACLE per-stretch upper bound\n(exposure FIXED; NOT GreenDyGNN's realized lag-bearing gain)")
    ax1.set_xlabel("congestion timescale  (epochs per stretch, exposure held fixed)")
    ax1.set_ylabel("gain (%)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.axhspan(0, 1.0, color="gray", alpha=0.08)
    ax2.axhline(3.0, ls="--", c="green", lw=1, label="STRONG (3%)")
    ax2.set_title("(b) NOVEL gain (per-stretch->DP), leak-guarded\nOptiSched's irreducible gain — should be FLAT & >0")
    ax2.set_xlabel("congestion timescale  (epochs per stretch, exposure held fixed)")
    ax2.set_ylabel("gain (%)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    print(f"  figure -> {path}")


def main():
    ap = argparse.ArgumentParser(description="OptiSched-GNN Phase-0 gate")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate demo traces instead of loading real ones")
    ap.add_argument("--traces", default="", help="root dir of real epoch traces")
    ap.add_argument("--part", type=int, default=0, help="which worker's traces to read")
    ap.add_argument("--owner_correlation", type=float, default=0.85)
    ap.add_argument("--datasets", default="reddit,ogbn-products,ogbn-papers100M")
    ap.add_argument("--model", default="", help="CostModel JSON (else defaults)")
    ap.add_argument("--num_partitions", type=int, default=4)
    ap.add_argument("--n_hot", type=int, default=2000)
    ap.add_argument("--w_max", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30, help="synthetic epochs")
    ap.add_argument("--stretch_lens", default="2,3,5,7,10,15",
                    help="congestion timescales to sweep (epochs per stretch)")
    ap.add_argument("--allocation", action="store_true",
                    help="enable per-owner allocation templates (joint opt)")
    ap.add_argument("--out", default="results/gate_results.json")
    ap.add_argument("--fig", default="../figures/fig_gate.pdf")
    args = ap.parse_args()

    use_synthetic = args.synthetic or not args.traces
    if args.model:
        model = CostModel.load(args.model)
    elif use_synthetic:
        # Demo model tuned to the synthetic scale so the rebuild/staleness
        # trade-off has an interior optimum (Theorem 4) and congestion shifts it;
        # real runs pass a profiled --model.  n_hot defaults to the matched 700.
        model = CostModel(num_partitions=args.num_partitions, w_max=args.w_max,
                          a=0.3, b=0.3, c=0.5, alpha=0.35,
                          t_miss=np.full(args.num_partitions, 4e-4))
        if args.n_hot == 2000:
            args.n_hot = 700
    else:
        model = default_model(num_partitions=args.num_partitions, w_max=args.w_max)
    print(f"Cost model: a={model.a:.4g} b={model.b:.4g} c={model.c:.3g} "
          f"alpha={model.alpha:.3g} P_bar={model.p_bar} t_miss={model.t_miss}")

    datasets = [d for d in args.datasets.split(",") if d]
    if use_synthetic:
        print("Source: SYNTHETIC demo traces")
        traces_by = make_synthetic(args.epochs, args.num_partitions)
    else:
        print(f"Source: real traces under {args.traces}")
        traces_by = load_real_traces(args.traces, datasets, part=args.part)

    templates = None
    if args.allocation:
        templates = make_templates(args.num_partitions, local_rank=0)

    stretch_lens = [int(s) for s in args.stretch_lens.split(",") if s]
    curves_by, diags, floors = {}, {}, {}
    for ds, traces in traces_by.items():
        print(f"\n=== gate sweep: {ds} ({len(traces)} epochs, "
              f"B={traces[0].num_batches}) ===")
        curve = G.sweep_dataset(ds, traces, model, args.n_hot, w_max=args.w_max,
                                templates=templates, stretch_lens=stretch_lens)
        curves_by[ds] = curve
        # FLOOR / near-floor decomposition logged BESIDE the novel-gain curves
        fr = G.floor_report(traces, model, args.n_hot, w_max=args.w_max)
        floors[ds] = fr
        print(f"  FLOOR(Thm A/F): L0(clean)={fr['clean']['L0']:.4g}J  "
              f"n*={fr['n_star_mean']:.0f}  beta8={fr['beta8']} beta32={fr['beta32']}")
        for k in ("clean", "cong"):
            f = fr[k]
            print(f"    {k:5s}: rho={f['rho']:.3f} (>=1 sandwich)  gamma_R={f['gamma_R']:.2f}  "
                  f"phi={f['phi']:.2f}  eta={f['eta']:.3g}  "
                  f"{'initiation-dominated' if f['initiation_dominated'] else 'payload-first'}")
        # congestion EXPOSURE (multiset of per-epoch sigma) -- fixed across the sweep
        print(f"  exposure (fixed across timescale): {curve[0].exposure}")
        # per-dataset curve table. NOTE: 'temporalUB' is the ORACLE per-stretch upper
        # bound at fixed exposure, NOT GreenDyGNN's realized (lag-bearing) gain.
        hdr = ("stretch", "#str", "leak", "static_kJ", "stretch_kJ", "perEp_kJ",
               "DP_kJ", "temporalUB%", "pStr->pEp%", "NOVEL%", "winEp%",
               "acrossHet", "withinHet")
        rows = [hdr]
        for r in curve:
            rows.append((str(r.stretch_len), str(r.n_stretches), str(r.leakage_flags),
                         f"{r.global_static_kJ:.1f}", f"{r.per_stretch_kJ:.1f}",
                         f"{r.per_epoch_kJ:.1f}", f"{r.dp_kJ:.1f}",
                         f"{r.temporal_gain_pct:.2f}", f"{r.per_stretch_vs_per_epoch_pct:.2f}",
                         f"{r.novel_gain_pct:.2f}", f"{r.within_epoch_gain_pct:.2f}",
                         f"{r.across_stretch_het:.2f}", f"{r.within_stretch_het:.2f}"))
        w = [max(len(rw[i]) for rw in rows) for i in range(len(hdr))]
        for k, rw in enumerate(rows):
            print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(rw)))
            if k == 0:
                print("  " + "  ".join("-" * w[i] for i in range(len(w))))
        d = G.curve_diagnostics(curve)
        diags[ds] = d
        print(f"  -> temporal(ORACLE UB, exposure-fixed) "
              f"{'STABLE' if d['temporal_not_increasing'] else 'RISING(!)'} "
              f"mean={d['temporal_mean']} {d['temporal_oracle_ub']}  "
              f"[= upper bound on any congestion-reactive uniform ctrl; NOT GreenDyGNN's "
              f"realized gain, which has reaction lag]")
        if d["baselines_collapsed_per_stretch_eq_per_epoch"]:
            print(f"  -> per-stretch == per-epoch (square-wave collapse to 3 baselines); "
                  f"pStr->pEp {d['per_stretch_vs_per_epoch']} ~0")
        else:
            print(f"  -> per-stretch != per-epoch (within-stretch trace variation!): "
                  f"pStr->pEp {d['per_stretch_vs_per_epoch']} — four baselines distinct, "
                  f"novel is the strict leak-guarded number")
        print(f"  -> novel    {'FLAT' if d['novel_flat'] else 'NOT-FLAT(!)'} "
              f"mean={d['novel_mean']} spread={d['novel_spread']} "
              f"rise(short->long)={d['novel_rise_short_to_long']} {d['novel']}")
        print(f"  -> within-epoch (timescale-invariant) {d['within_epoch']}")
        if d["leakage_total"]:
            print(f"  !! LEAKAGE {d['leakage_total']} — per-stretch baseline mixed "
                  f"congestion levels; novel gap is INVALID")

    dec = G.decision_from_curves(diags)
    pred = G.predictor_pooled(curves_by)
    print("\n" + "=" * 70)
    print("PRE-REGISTERED DECISION (see GATE_PREREGISTRATION.md):", dec["verdict"])
    print(f"  datasets clearing (novel>=3% & flat & no-leak): {dec['n_clearing']} "
          f"-> {dec['per_dataset']}")
    print(f"PREDICTOR (Rule 2): pooled Spearman(within-het, novel) over {pred['n_points']} "
          f"points = {pred['pooled_spearman_within_het_vs_novel']}  "
          f"(3pt cross-dataset = {pred['cross_dataset_spearman_3pt']}; "
          f"headline rests on the POOLED number)")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "model": model.to_dict(), "n_hot": args.n_hot, "w_max": args.w_max,
        "allocation": args.allocation, "stretch_lens": stretch_lens,
        "provenance": provenance(),
        "decision": dec, "predictor": pred, "diagnostics": diags,
        "floor": floors,
        "curves": {ds: [vars(r) for r in curve] for ds, curve in curves_by.items()},
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else o)
    print(f"\nresults -> {args.out}")
    if args.fig:
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        make_figure(curves_by, args.fig)


if __name__ == "__main__":
    main()
