#!/usr/bin/env python3
"""EXPLORATORY predictor search (NOT pre-registered).

The pre-registered predictor -- within-stretch W*(t) variance -> novel gain --
was FALSIFIED on real traces (pooled Spearman -0.52; Products has 45x lower
within-het yet higher gain).  This script is the follow-up *finding*: across all
real epoch-traces, which structural trace property actually rank-orders the
per-epoch novel gain?  Candidate drivers (footprint/cache pressure, reuse,
payload) are scored against BOTH the (default, untrustworthy) time-model novel
gain AND the floor headroom rho-1, and compared head-to-head with within-het.

Exploratory only: 2 datasets x 8 epochs = 16 points, default cost model.  Re-run
under calibrated energy before drawing conclusions.
"""
import argparse, glob, os, numpy as np
from optisched.trace import Trace
from optisched.calibration import CostModel
from optisched import interval_cost as IC, dp_solver as DP, floor as FL

D_BY_DATASET = {"reddit": 602, "ogbn-products": 100, "ogbn-papers100M": 128}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return 0.0
    rx = x.argsort().argsort().astype(float); ry = y.argsort().argsort().astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def per_epoch_metrics(tr, model, n_hot, w_max):
    fp = FL.footprints(tr)
    Um = fp["U_m"]; U = fp["U"]; A = fp["A"]
    ic = IC.precompute(tr, model, n_hot, w_max=w_max)
    cost, bt = ic.cost_matrix(model, np.ones(tr.num_partitions), "A")
    pred = {
        "within_het(FAILED)": float(np.var(DP.per_position_w(cost, w_max))),
        "|U|/n_hot": U / n_hot,
        "max|Um|/n_hot": float(Um.max()) / n_hot,
        "reuse A/|U|": A / max(1.0, U),
        "beta8": FL.working_set_drift(tr, 8),
        "mean_remote/batch": A / max(1, tr.num_batches),
    }
    dp = DP.solve_optionA(cost, bt, w_max)
    grid = [w for w in (1, 2, 4, 8, 16) if w <= w_max]
    _, per_w = DP.oracle_uniform(cost, grid, bt)
    best_u = min(per_w.values())
    novel_time = 100.0 * (best_u - dp.cost) / best_u if best_u > 0 else 0.0
    fl = FL.communication_floor(tr, model)
    tf = FL.schedule_transfers(tr, dp.windows, model, n_hot, rebuild="delta")
    rho = FL.near_floor(fl, tf, model).rho
    return pred, {"novel_time%": novel_time, "rho-1(floor)": rho - 1.0}


def main(a):
    rows = []
    for ds in a.datasets.split(","):
        files = sorted(glob.glob(os.path.join(a.traces, ds, "r0_epoch_*.npz")))
        if not files:
            print(f"[warn] no traces for {ds}"); continue
        model = CostModel(num_partitions=4, w_max=a.w_max, d=D_BY_DATASET.get(ds, 128))
        for f in files:
            tr = Trace.load(f)
            pred, tgt = per_epoch_metrics(tr, model, a.n_hot, a.w_max)
            rows.append({"ds": ds, **pred, **tgt})
        print(f"{ds}: {len(files)} epochs (d={D_BY_DATASET.get(ds,128)})")

    if not rows:
        return
    preds = [k for k in rows[0] if k not in ("ds", "novel_time%", "rho-1(floor)")]
    print(f"\n{'='*72}\nEXPLORATORY predictor ranking ({len(rows)} epoch-points, "
          f"default model)\n{'='*72}")
    for target in ("novel_time%", "rho-1(floor)"):
        print(f"\nTarget = {target}:")
        scored = sorted(preds, key=lambda p: -abs(spearman([r[p] for r in rows],
                                                            [r[target] for r in rows])))
        for p in scored:
            s = spearman([r[p] for r in rows], [r[target] for r in rows])
            print(f"   Spearman = {s:+.3f}   {p}")
    print(f"\n{'-'*72}\nper-dataset means:")
    for ds in sorted(set(r["ds"] for r in rows)):
        sub = [r for r in rows if r["ds"] == ds]
        mean = lambda k: float(np.mean([r[k] for r in sub]))
        print(f"  {ds:16s} |U|/n_hot={mean('|U|/n_hot'):.2f}  reuse={mean('reuse A/|U|'):.2f}  "
              f"within_het={mean('within_het(FAILED)'):.2f}  "
              f"novel_time={mean('novel_time%'):.1f}%  rho-1={mean('rho-1(floor)'):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="traces")
    p.add_argument("--datasets", default="reddit,ogbn-products")
    p.add_argument("--n_hot", type=int, default=100000)
    p.add_argument("--w_max", type=int, default=20)
    main(p.parse_args())
