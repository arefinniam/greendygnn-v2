#!/usr/bin/env python3
"""LOAD-BEARING experiment: joint owner-aware allocation x per-owner window
scheduling, in CALIBRATED payload/refetch energy.

Question: when cache is limited, does the full architecture (greedy/template owner
allocation + per-owner energy-optimal window DP) reduce payload inflation phi and
floor distance rho MORE than allocation-alone or windowing-alone?

Method matrix (x cache pressure n_hot/|U| in {0.25,0.5,0.75,1.0,1.5}):
  1 uniform alloc      + fixed W=16
  2 uniform alloc      + global W-DP (one energy-DP boundary seq, all owners)
  3 uniform alloc      + per-owner W-DP (energy)
  4 greedy alloc       + fixed W=16
  5 greedy alloc       + per-owner W-DP
  6 template alloc     + per-owner W-DP   (best static split)
Everything evaluated by ACTUAL delta-rebuild transfers (floor.schedule_transfers):
report rho=E/L0, phi=Q/|U|, calibrated energy.
"""
import argparse, glob, json, os, numpy as np
from optisched.trace import Trace
from optisched.calibration import CostModel
from optisched import interval_cost as IC, dp_solver as DP, floor as FL


def uniform_windows(B, W):
    return [(s, min(W, B - s)) for s in range(0, B, W)]


def energy_cost_matrix(ic, model, m):
    """Self-contained calibrated-energy interval cost for owner m (full-refresh
    rebuild rows + residual miss rows), for the per-owner energy W-DP."""
    B, Wm = ic.B, ic.W_max
    cap = max(model.c_max[m], 1.0)
    hot = ic.hot[:, :, 0, m]; res = ic.residual[:, :, 0, m]
    rows = hot + res
    rpcs = np.ceil(hot / cap) + np.ceil(np.where(np.isfinite(res), res, 0) / cap)
    E = model.kappa[m] * model.d * rows + model.eps_init[m] * rpcs
    E = np.where(ic.valid, E, np.inf)
    return E


def owner_windows(sub, model, m, n_m, W, policy, global_wins=None):
    if policy == "fixedW":
        return uniform_windows(sub.num_batches, W)
    if policy == "global":
        return global_wins
    # per-owner energy W-DP
    ic = IC.precompute(sub, model, int(max(1, n_m)), w_max=W)
    E = energy_cost_matrix(ic, model, m)
    return DP.solve_optionA(E, None, w_max=W).windows


def owner_energy(sub, model, m, n_m, windows):
    tf = FL.schedule_transfers(sub, windows, model, int(max(1, n_m)), rebuild="delta")
    E = model.eps_init[m] * tf["R_m"][m] + model.kappa[m] * model.d * tf["Q_m"][m]
    return E, tf["Q_m"][m]


def main(a):
    files = sorted(glob.glob(os.path.join(a.traces, a.dataset, "r0_epoch_*.npz")))[:a.epochs]
    model = CostModel.load(a.model)
    traces = [Trace.load(f) for f in files]
    P = traces[0].num_partitions
    owners = [m for m in range(P) if m != traces[0].local_rank]
    U = float(np.mean([FL.footprints(t)["U"] for t in traces]))
    Um = {m: float(np.mean([FL.footprints(t)["U_m"][m] for t in traces])) for m in owners}
    Am = {m: float(np.mean([t.restrict_owner(m).nodes.size for t in traces])) for m in owners}
    L0 = float(np.mean([FL.communication_floor(t, model).L0 for t in traces]))
    Wfix = a.W
    print(f"{a.dataset}: {len(traces)} ep, B={traces[0].num_batches}, |U|={U:.0f}, "
          f"|U_m|={ {m:int(v) for m,v in Um.items()} }, d={model.d}")

    def greedy_alloc(n_hot):
        # marginal-greedy using fixed-W16 energy curves (cheap, decides the split)
        grid = np.unique(np.linspace(0, n_hot, 21).astype(int))
        curve = {m: np.zeros(len(grid)) for m in owners}
        for m in owners:
            for tr in traces:
                sub = tr.restrict_owner(m)
                w = uniform_windows(sub.num_batches, Wfix)
                for gi, nm in enumerate(grid):
                    curve[m][gi] += owner_energy(sub, model, m, nm, w)[0]
        block = max(1, n_hot // 20); alloc = {m: 0 for m in owners}; spent = 0
        while spent + block <= n_hot:
            bm, bg = None, 1e-30
            for m in owners:
                i0 = int(np.argmin(abs(grid - alloc[m]))); i1 = int(np.argmin(abs(grid - (alloc[m] + block))))
                g = curve[m][i0] - curve[m][i1]
                if g > bg: bg, bm = g, m
            if bm is None: break
            alloc[bm] += block; spent += block
        return {m: min(alloc[m], Um[m]) for m in owners}

    def eval_method(alloc, policy, n_hot):
        E_tot, Q_tot = 0.0, 0.0; phim = {}
        for m in owners:
            n_m = alloc[m]
            # global energy-DP boundaries (computed on full trace per epoch) reused per owner
            E_m_sum, Q_m_sum = 0.0, 0.0
            for tr in traces:
                sub = tr.restrict_owner(m)
                gw = None
                if policy == "global":
                    icf = IC.precompute(tr, model, int(max(1, n_hot)), w_max=Wfix)
                    Ef = sum(energy_cost_matrix(icf, model, mm) for mm in owners)
                    Ef = np.where(icf.valid, Ef, np.inf)
                    gw = DP.solve_optionA(Ef, None, w_max=Wfix).windows
                w = owner_windows(sub, model, m, n_m, Wfix, policy, gw)
                e, q = owner_energy(sub, model, m, n_m, w)
                E_m_sum += e; Q_m_sum += q
            phim[m] = Q_m_sum / max(1.0, Um[m] * len(traces))
            E_tot += E_m_sum; Q_tot += Q_m_sum
        rho = E_tot / (L0 * len(traces)) if L0 > 0 else float("inf")
        phi = Q_tot / max(1.0, U * len(traces))
        return {"rho": rho, "phi": phi, "E": E_tot,
                "phi_m": {int(m): round(phim[m], 3) for m in owners}}

    rows = []
    for r in a.ratios:                       # r = n_hot/|U|
        n_hot = max(len(owners), int(round(r * U)))
        unif = {m: n_hot / len(owners) for m in owners}
        greedy = greedy_alloc(n_hot)
        # template = best static split among {uniform, propU, propA, greedy} by fixedW
        cand = {"uniform": unif,
                "propU": {m: n_hot * Um[m] / sum(Um.values()) for m in owners},
                "propA": {m: n_hot * Am[m] / sum(Am.values()) for m in owners},
                "greedy": greedy}
        tmpl = min(cand.values(), key=lambda al: eval_method(al, "fixedW", n_hot)["E"])
        methods = {
            "1 unif+W16":        (unif, "fixedW"),
            "2 unif+globalDP":   (unif, "global"),
            "3 unif+ownerDP":    (unif, "perowner"),
            "4 greedy+W16":      (greedy, "fixedW"),
            "5 greedy+ownerDP":  (greedy, "perowner"),
            "6 template+ownerDP":(tmpl, "perowner"),
        }
        print(f"\n=== {a.dataset}  pressure |U|/n_hot = {U/n_hot:.2f}  (n_hot={n_hot}) ===")
        print(f"{'method':>20} {'rho':>7} {'phi':>7} {'E':>11}")
        res = {}
        for name, (al, pol) in methods.items():
            v = eval_method(al, pol, n_hot)
            res[name] = v
            print(f"{name:>20} {v['rho']:>7.3f} {v['phi']:>7.3f} {v['E']:>11.4g}")
        rows.append({"pressure": U / n_hot, "n_hot": n_hot, "methods": res,
                     "greedy_split": {int(m): int(greedy[m]) for m in owners}})
    out = {"dataset": a.dataset, "U": U, "L0": L0, "Um": {int(m): Um[m] for m in owners},
           "rows": rows, "note": "joint allocation x per-owner window DP, calibrated energy"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="traces"); p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True); p.add_argument("--W", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5])
    p.add_argument("--out", default="results/joint_ablation.json")
    main(p.parse_args())
