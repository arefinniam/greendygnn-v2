#!/usr/bin/env python3
"""Cache-pressure ablation (MECHANISM test, NOT the pre-registered gate).

The pre-registered gate produced no final verdict (original predictor failed;
calibration changed the interpretation).  This ablation identifies the operating
regime: sweep the cache budget and measure where the scheduling/caching advantage
emerges, with the true x-axis being cache pressure |U|/n_hot.

For each dataset and each n_hot, on the REAL calibrated traces:
  x  = |U| / n_hot                         (cache pressure)
  y1 = calibrated novel gain (DP vs oracle-uniform, time model)
  y2 = rho = E_DP / L0                     (floor distance; P_bar-invariant)
  y3 = phi = Q_DP / |U|                    (payload inflation / refetch)

Expected phase transition: pressure ≈1 -> rho≈1, phi≈1, gain≈0; pressure>1 ->
rho,phi rise and the gain appears; very high pressure -> gain may plateau/shrink.
"""
import argparse, glob, json, os, numpy as np
from optisched.trace import Trace
from optisched.calibration import CostModel
from optisched import interval_cost as IC, dp_solver as DP, floor as FL


def main(a):
    files = sorted(glob.glob(os.path.join(a.traces, a.dataset, "r0_epoch_*.npz")))
    if not files:
        print(f"[warn] no traces for {a.dataset}"); return
    files = files[:a.epochs]
    model = CostModel.load(a.model)
    traces = [Trace.load(f) for f in files]
    U_mean = float(np.mean([FL.footprints(t)["U"] for t in traces]))
    ratios = [float(x) for x in a.nhot_over_U.split(",")]   # n_hot / |U|
    grid = [w for w in (1, 2, 4, 8, 16) if w <= a.w_max]
    print(f"{a.dataset}: {len(traces)} epochs, B={traces[0].num_batches}, "
          f"mean|U|={U_mean:.0f}, d={model.d}")

    rows = []
    for r in ratios:
        n_hot = max(1, int(round(r * U_mean)))
        novel, rho, phi = [], [], []
        for tr in traces:
            ic = IC.precompute(tr, model, n_hot, w_max=a.w_max)
            cost, bt = ic.cost_matrix(model, np.ones(tr.num_partitions), "A")
            dp = DP.solve_optionA(cost, bt, a.w_max)
            _, per_w = DP.oracle_uniform(cost, grid, bt)
            bu = min(per_w.values())
            novel.append(100.0 * (bu - dp.cost) / bu if bu > 0 else 0.0)
            fl = FL.communication_floor(tr, model)
            tf = FL.schedule_transfers(tr, dp.windows, model, n_hot, rebuild="delta")
            nf = FL.near_floor(fl, tf, model)
            rho.append(nf.rho); phi.append(nf.phi)
        pressure = U_mean / n_hot
        rows.append({"nhot_over_U": r, "n_hot": n_hot, "pressure_U_over_nhot": pressure,
                     "novel_pct": float(np.mean(novel)), "rho": float(np.mean(rho)),
                     "phi": float(np.mean(phi))})

    rows.sort(key=lambda x: x["pressure_U_over_nhot"])
    print(f"\n{'pressure |U|/n_hot':>18} {'n_hot':>8} {'novel%':>8} {'rho':>7} {'phi':>7}")
    print("-" * 52)
    for x in rows:
        print(f"{x['pressure_U_over_nhot']:>18.2f} {x['n_hot']:>8d} "
              f"{x['novel_pct']:>8.2f} {x['rho']:>7.3f} {x['phi']:>7.3f}")
    out = {"dataset": a.dataset, "mean_U": U_mean, "d": model.d,
           "calibration_basis": model.to_dict().get("_calibration", "n/a"),
           "note": "MECHANISM ABLATION, not the pre-registered gate", "sweep": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="traces")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--w_max", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--nhot_over_U", default="0.1,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--out", default="results/cache_sweep.json")
    main(p.parse_args())
