#!/usr/bin/env python3
"""Owner-aware cache-allocation ablation in CALIBRATED payload/refetch energy.

The cache-pressure ablation showed the energy gap is payload/refetch driven by
cache pressure, and global-W scheduling does not close it.  This experiment tests
the reframed lever (Architecture Thm G + Thm B): given a fixed TOTAL cache budget,
how to SPLIT it across owners and refresh each owner's cache to minimise the
calibrated communication energy E = sum_m [eps_init*R_m + kappa*d*Q_m].

Per owner m: build the sub-trace, tile it with windows of length W (delta rebuild,
weighted top-frequency hot nodes), and measure E_m(n_m) = energy as a function of
that owner's budget.  Then compare allocation strategies:
  uniform        n_m = n_hot/(P-1)
  prop_|U_m|     n_m proportional to per-owner footprint
  prop_A_m       n_m proportional to per-owner access volume
  marginal-greedy  allocate each cache block to the owner with the largest
                   marginal energy reduction dE_m  (the principled rule; optimal
                   for separable convex-decreasing E_m)
Baseline: a single shared cache run globally (no owner split).  Report E and the
floor distance rho = E / L0 for each.
"""
import argparse, glob, json, os, numpy as np
from optisched.trace import Trace
from optisched.calibration import CostModel
from optisched import floor as FL


def uniform_windows(B, W):
    return [(s, min(W, B - s)) for s in range(0, B, W)]


def owner_energy_curve(sub, model, W, budgets, owner_id):
    """E_m(n_m) for a grid of budgets, using delta-rebuild windowed hot sets."""
    wins = uniform_windows(sub.num_batches, W)
    E = []
    for nm in budgets:
        tf = FL.schedule_transfers(sub, wins, model, int(max(1, nm)), rebuild="delta")
        # owner m's calibrated energy (rows + RPCs for this owner)
        Em = model.eps_init[owner_id] * tf["R_m"][owner_id] + \
             model.kappa[owner_id] * model.d * tf["Q_m"][owner_id]
        E.append(Em)
    return np.array(E)


def evaluate_split(curves, budgets, alloc):
    """Total energy for a budget split alloc={m: n_m} via nearest grid point."""
    tot = 0.0
    for m, nm in alloc.items():
        idx = int(np.argmin(np.abs(budgets - nm)))
        tot += curves[m][idx]
    return tot


def main(a):
    files = sorted(glob.glob(os.path.join(a.traces, a.dataset, "r0_epoch_*.npz")))[:a.epochs]
    model = CostModel.load(a.model)
    traces = [Trace.load(f) for f in files]
    P = traces[0].num_partitions
    owners = [m for m in range(P) if m != traces[0].local_rank]
    n_hot = a.n_hot
    budgets = np.unique(np.linspace(0, n_hot, a.grid).astype(int))
    print(f"{a.dataset}: {len(traces)} epochs, total n_hot={n_hot}, W={a.W}, d={model.d}")

    # accumulate per-owner energy curves + footprints across epochs
    curves = {m: np.zeros(len(budgets)) for m in owners}
    Um = {m: 0.0 for m in owners}; Am = {m: 0.0 for m in owners}; L0 = 0.0
    for tr in traces:
        L0 += FL.communication_floor(tr, model).L0
        fp = FL.footprints(tr)
        for m in owners:
            Um[m] += fp["U_m"][m]
            sub = tr.restrict_owner(m)
            Am[m] += sub.nodes.size
            curves[m] += owner_energy_curve(sub, model, a.W, budgets, m)

    # allocation strategies
    def cap(alloc):  # clamp to footprint, never exceed total
        return {m: min(alloc[m], Um[m]) for m in owners}
    sU = sum(Um.values()); sA = sum(Am.values())
    strategies = {
        "uniform":     {m: n_hot / len(owners) for m in owners},
        "prop_|U_m|":  {m: n_hot * Um[m] / sU for m in owners},
        "prop_A_m":    {m: n_hot * Am[m] / sA for m in owners},
    }
    # marginal-greedy: blocks to the owner with largest marginal dE per block
    block = max(1, n_hot // (a.grid - 1))
    galloc = {m: 0 for m in owners}; spent = 0
    while spent + block <= n_hot:
        best_m, best_gain = None, 0.0
        for m in owners:
            i0 = int(np.argmin(np.abs(budgets - galloc[m])))
            i1 = int(np.argmin(np.abs(budgets - (galloc[m] + block))))
            gain = curves[m][i0] - curves[m][i1]   # energy reduction
            if gain > best_gain:
                best_gain, best_m = gain, m
        if best_m is None:
            break
        galloc[best_m] += block; spent += block
    strategies["marginal-greedy"] = galloc

    print(f"\n{'strategy':>16} {'E (calib)':>12} {'rho=E/L0':>10}  split (n_m by owner)")
    print("-" * 78)
    results = {}
    for name, alloc in strategies.items():
        alloc = cap(alloc)
        E = evaluate_split(curves, budgets, alloc)
        rho = E / L0 if L0 > 0 else float("inf")
        results[name] = {"E": float(E), "rho": float(rho),
                         "split": {int(m): int(round(alloc[m])) for m in owners}}
        print(f"{name:>16} {E:12.4g} {rho:10.3f}  "
              f"{ {int(m): int(round(alloc[m])) for m in owners} }")
    base = results["uniform"]["E"]
    print(f"\nmarginal-greedy vs uniform: "
          f"{100*(base-results['marginal-greedy']['E'])/base:+.2f}% energy")
    print(f"per-owner footprint |U_m|: { {int(m): int(Um[m]) for m in owners} }")
    out = {"dataset": a.dataset, "n_hot": n_hot, "W": a.W, "L0": L0,
           "U_m": {int(m): float(Um[m]) for m in owners},
           "A_m": {int(m): float(Am[m]) for m in owners},
           "strategies": results,
           "note": "owner-aware allocation ablation in calibrated payload/refetch energy"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="traces")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n_hot", type=int, default=100000)
    p.add_argument("--W", type=int, default=16)
    p.add_argument("--grid", type=int, default=41)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--out", default="results/alloc_ablation.json")
    main(p.parse_args())
