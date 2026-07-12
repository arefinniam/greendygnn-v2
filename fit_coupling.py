#!/usr/bin/env python3
"""fit_coupling.py — fit the overlap-aware coupling parameters of StepModel.

Fits ONLY the seven couplings (t_c, alpha_reb, phi_miss, phi_reb, phi_delay,
r0_delay, beta_ov) per dataset, on the FIT cells only:
  * clean W-sweep   ({ds}_clean_W{1..128})
  * severity @ W16  ({ds}_c1_{rate}_W16, {ds}_c4_{ms}ms_W16)
Holdout cells (*_vg_*) are NEVER read here — compare_sim_real.py evaluates them.

All calibrated primitives (h(W), U(W), T_rebuild(a,b,c), per-row costs, R,
reuse) are taken from the existing data/calib_<ds>.json unchanged; the fitted
couplings are written back into that JSON (plus a _coupling_fit provenance
block). Objective: mean squared log-ratio log(pred/meas)^2 (balances ms-scale
clean cells against second-scale congested cells). Optimizer: seeded random
search + shrinking coordinate refinement (deterministic, no scipy).

  python3 fit_coupling.py --calib_runs ~/greendygnn_work/calib_runs \
      --datasets reddit,ogbn-products
"""
import argparse
import glob
import json
import math
import os

import numpy as np

from simulator import CalibParams, StepModel
from compare_sim_real import KAPPA_NB, measured_step, cell_condition

BOUNDS = {
    "t_c":       (1e-3, 0.30),
    "alpha_reb": (0.0, 2.0),
    "phi_miss":  (0.0, 1.0),
    "phi_reb":   (0.0, 1.0),
    "phi_delay": (0.0, 2.0),
    "r0_delay":  (0.0, 20.0),
    "beta_ov":   (0.0, 3.0),
}
NAMES = list(BOUNDS)


def fit_cells(calib_runs: str, ds: str):
    cells = []
    for d in sorted(glob.glob(os.path.join(calib_runs, f"{ds}_clean_W*"))):
        cells.append(("clean", int(d.rsplit("W", 1)[-1]), d))
    for d in sorted(glob.glob(os.path.join(calib_runs, f"{ds}_c[14]_*_W16"))):
        cond = d.split(f"{ds}_")[-1].rsplit("_W16", 1)[0]
        cells.append((cond, 16, d))
    out = []
    for cond, W, d in cells:
        meas = measured_step(d)
        if meas is None:
            continue
        kappa, delta = cell_condition(cond) if cond != "clean" \
            else (np.ones(3), np.zeros(3))
        out.append({"cond": cond, "W": W, "kappa": kappa, "delta": delta,
                    "meas": meas})
    return out


def make_objective(base: CalibParams, cells):
    def obj(theta):
        p = CalibParams(**{**base.__dict__,
                           **{n: float(v) for n, v in zip(NAMES, theta)}})
        # static runs used uniform allocation
        m = StepModel(p, alloc_on=False)
        s = 0.0
        for c in cells:
            pred, _ = m.step_time(c["W"], c["kappa"], c["delta"])
            s += math.log(max(pred, 1e-6) / c["meas"]) ** 2
        return s / len(cells)
    return obj


def optimize(obj, seed=0, n_random=4000, n_refine=6000):
    rng = np.random.default_rng(seed)
    lo = np.array([BOUNDS[n][0] for n in NAMES])
    hi = np.array([BOUNDS[n][1] for n in NAMES])
    best_x, best_f = None, math.inf
    for _ in range(n_random):
        x = lo + rng.random(len(NAMES)) * (hi - lo)
        f = obj(x)
        if f < best_f:
            best_x, best_f = x.copy(), f
    step = (hi - lo) * 0.10
    for it in range(n_refine):
        j = it % len(NAMES)
        for sgn in (+1, -1):
            x = best_x.copy()
            x[j] = float(np.clip(x[j] + sgn * step[j] * rng.random(),
                                 lo[j], hi[j]))
            f = obj(x)
            if f < best_f:
                best_x, best_f = x, f
                break
        if it and it % (len(NAMES) * 60) == 0:
            step *= 0.7
    return best_x, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_runs", required=True)
    ap.add_argument("--datasets", default="reddit,ogbn-products")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for ds in args.datasets.split(","):
        path = f"data/calib_{ds}.json"
        with open(path) as f:
            raw = json.load(f)
        base = CalibParams.from_dict(raw)
        cells = fit_cells(args.calib_runs, ds)
        assert cells, f"no fit cells found for {ds} in {args.calib_runs}"
        assert not any("vg" in c["cond"] for c in cells)
        obj = make_objective(base, cells)
        theta, f = optimize(obj, seed=args.seed)
        fitted = {n: round(float(v), 6) for n, v in zip(NAMES, theta)}
        print(f"\n=== {ds} ===  msle={f:.5f}")
        print(json.dumps(fitted, indent=1))

        # per-cell fit errors
        p = CalibParams(**{**base.__dict__, **fitted})
        m = StepModel(p, alloc_on=False)
        errs = []
        for c in cells:
            pred, _ = m.step_time(c["W"], c["kappa"], c["delta"])
            e = (pred - c["meas"]) / c["meas"] * 100
            errs.append(abs(e))
            print(f"  {c['cond']:>12s} W{c['W']:<4d} meas {c['meas']*1e3:8.1f} "
                  f"pred {pred*1e3:8.1f}  {e:+6.1f}%")
        print(f"  mean|err| on FIT cells: {np.mean(errs):.1f}%")

        raw.update(fitted)
        raw["_coupling_fit"] = {"msle": round(f, 6),
                                "mean_abs_err_fit_pct": round(float(np.mean(errs)), 1),
                                "n_fit_cells": len(cells), "seed": args.seed,
                                "bounds": {n: list(BOUNDS[n]) for n in NAMES}}
        with open(path, "w") as fo:
            json.dump(raw, fo, indent=1)
        print(f"  wrote couplings into {path}")


if __name__ == "__main__":
    main()
