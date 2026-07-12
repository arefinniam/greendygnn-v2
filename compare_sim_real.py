#!/usr/bin/env python3
"""compare_sim_real.py — the REAL simulator-validation grid (replaces the
synthesized Fig 8). Compares StepModel-predicted step time against measured
steady-state step time for every calibration cell AND the holdout cells that
were never used in fitting (validate_grid.sh: W in {4,8,32} x {c1_200mbit,
c1_50mbit, c4_10ms}).

  python3 compare_sim_real.py --calib_runs ~/greendygnn_work/calib_runs \
      --datasets reddit,ogbn-products --out simreal_grid.json

Ground-truth kappa per rate from the verification canary (netbench, measured
2026-07-02): 1000mbit=9.91 500mbit=19.37 200mbit=48.42 100mbit=96.84 50mbit=193.66.
"""
import argparse, glob, json, os
import numpy as np
from simulator import CalibParams, StepModel

KAPPA_NB = {"1000mbit": 9.91, "500mbit": 19.37, "200mbit": 48.42,
            "100mbit": 96.84, "50mbit": 193.66}


def measured_step(run_dir, warmup=2):
    ts = []
    for pf in glob.glob(os.path.join(run_dir, "*_part*_profile.json")):
        with open(pf) as f:
            p = json.load(f)
        ss = [s["step_time_s"] for s in p.get("steps", [])
              if s.get("epoch", 0) >= warmup]
        if ss:
            ts.append(float(np.mean(ss)))
    return float(np.mean(ts)) if ts else None


def cell_condition(cond):
    """-> (kappa_vec, delta_vec) for victim = last remote slot."""
    kappa = np.ones(3); delta = np.zeros(3)
    if cond.startswith("c1_"):
        kappa[-1] = KAPPA_NB[cond[3:]]
    elif cond.startswith("c4_"):
        delta[-1] = float(cond[3:].replace("ms", "")) / 1e3
    return kappa, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_runs", required=True)
    ap.add_argument("--datasets", default="reddit,ogbn-products")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="simreal_grid.json")
    args = ap.parse_args()

    result = {}
    for ds in args.datasets.split(","):
        p = CalibParams.load(f"data/calib_{ds}.json")
        model = StepModel(p, alloc_on=False)   # static runs: no cost-aware alloc
        cells = []
        # calibration cells: clean x all W
        for d in sorted(glob.glob(os.path.join(args.calib_runs, f"{ds}_clean_W*"))):
            W = int(d.rsplit("W", 1)[-1])
            cells.append(("clean", W, d, "fit"))
        # calibration cells: severity x W16
        for d in sorted(glob.glob(os.path.join(args.calib_runs, f"{ds}_c[14]_*_W16"))):
            cond = d.split(f"{ds}_")[-1].rsplit("_W16", 1)[0]
            cells.append((cond, 16, d, "fit"))
        # holdout cells
        for d in sorted(glob.glob(os.path.join(args.calib_runs, f"{ds}_vg_*_W*"))):
            rest = d.split(f"{ds}_vg_")[-1]
            cond, wpart = rest.rsplit("_W", 1)
            cells.append((cond, int(wpart), d, "holdout"))

        rows, fit_errs, hold_errs = [], [], []
        for cond, W, d, kind in cells:
            meas = measured_step(d, args.warmup)
            if meas is None:
                continue
            kappa, delta = cell_condition(cond) if cond != "clean" else (np.ones(3), np.zeros(3))
            pred, _ = model.step_time(W, kappa, delta)
            err = (pred - meas) / meas * 100.0
            rows.append({"cond": cond, "W": W, "kind": kind,
                         "measured_ms": round(meas * 1e3, 2),
                         "predicted_ms": round(pred * 1e3, 2),
                         "err_pct": round(err, 1)})
            (fit_errs if kind == "fit" else hold_errs).append(abs(err))
        # best-W rank agreement per condition — the policy lives on rankings,
        # so this is validation target (i); uses every condition with >=3
        # distinct measured W cells (fit + holdout combined per condition).
        bycond = {}
        for r in rows:
            bycond.setdefault(r["cond"], []).append(r)
        rank_rows = []
        for cond, rs in sorted(bycond.items()):
            if len({r["W"] for r in rs}) < 3:
                continue
            rs = sorted(rs, key=lambda r: r["W"])
            meas = np.array([r["measured_ms"] for r in rs])
            pred = np.array([r["predicted_ms"] for r in rs])
            ws = [r["W"] for r in rs]

            def rankcorr(x, y):
                rx = np.argsort(np.argsort(x)).astype(float)
                ry = np.argsort(np.argsort(y)).astype(float)
                if rx.std() == 0 or ry.std() == 0:
                    return 1.0
                return float(np.corrcoef(rx, ry)[0, 1])

            rank_rows.append({
                "cond": cond, "Ws": ws,
                "best_w_measured": ws[int(np.argmin(meas))],
                "best_w_predicted": ws[int(np.argmin(pred))],
                "best_w_match": bool(int(np.argmin(meas)) == int(np.argmin(pred))),
                "spearman": round(rankcorr(meas, pred), 3),
            })
        result[ds] = {
            "cells": rows,
            "mean_abs_err_fit_pct": round(float(np.mean(fit_errs)), 1) if fit_errs else None,
            "mean_abs_err_holdout_pct": round(float(np.mean(hold_errs)), 1) if hold_errs else None,
            "n_fit": len(fit_errs), "n_holdout": len(hold_errs),
            "rank_agreement": rank_rows,
            "best_w_match_rate": round(float(np.mean(
                [r["best_w_match"] for r in rank_rows])), 3) if rank_rows else None,
        }
        print(f"\n=== {ds} ===")
        print(f"{'cond':>12s} {'W':>4s} {'kind':>8s} {'meas ms':>8s} {'pred ms':>8s} {'err%':>7s}")
        for r in rows:
            print(f"{r['cond']:>12s} {r['W']:4d} {r['kind']:>8s} "
                  f"{r['measured_ms']:8.1f} {r['predicted_ms']:8.1f} {r['err_pct']:7.1f}")
        print(f"mean|err| fit={result[ds]['mean_abs_err_fit_pct']}% "
              f"holdout={result[ds]['mean_abs_err_holdout_pct']}%")
        for rr in result[ds]["rank_agreement"]:
            print(f"  rank[{rr['cond']:>12s}] bestW meas={rr['best_w_measured']:<3d} "
                  f"pred={rr['best_w_predicted']:<3d} "
                  f"match={rr['best_w_match']} rho={rr['spearman']}")
        print(f"best-W match rate: {result[ds]['best_w_match_rate']}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
