#!/usr/bin/env python3
"""fit_calibration.py — turn calibrate_cluster.sh run profiles into a real
calibration JSON for simulator.CalibratedModel (replaces calib_synthetic.json).

Runs LOCALLY on profiles pulled from the cluster:
  python3 fit_calibration.py --calib_dir <dir with {ds}_clean_W*/, {ds}_c1_*/, {ds}_c4_*/>
                             --dataset reddit --out data/calib_reddit.json
                             [--warmup 2] [--report calib_report_reddit.md]

Estimators (documented per CALIBRATION_PROTOCOL.md):
  h(W)        logistic fit (Eq.2) on steady-state hit rates, bootstrap CIs,
              holdout: fit on W in {1,4,16,64}, validate on {2,8,32,128}.
  T_rebuild   a + b*W^c on per-window (t_plan + t_fetch), a >= 0.
  U(W)        U0 * W^u log-log fit on unique_remote_nodes per window.
  t_bulk_row  median(t_fetch / rows_fetched) over clean rebuilds.
  t_init/t_pay per-owner OLS of mean_rtt vs mean rows across owners+epochs (clean).
  R_per_batch total_remote_accesses / win_len (clean, all windows).
  reuse_frac  rows_reused / unique_remote_nodes (clean, steady).
  t_base      min over W of steady (step_time - rebuild_amortized - fetch_time).
  kappa(rate) victim mean_rtt_per_row / clean mean_rtt_per_row (c1 runs) —
              cross-checked against netbench verification JSON if provided.
  sigma axis  per-fetch additive delta from c4 delay runs.
  c_ar        residual step-time inflation under c4 beyond fetch inflation.
  power       per-epoch RAPL/NVML deltas (median of positive; wrap-robust).
"""
import argparse, glob, json, math, os, sys
import numpy as np

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


# ---------------------------------------------------------------- loading --
def load_parts(run_dir):
    parts = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*_part*_profile.json"))):
        with open(p) as f:
            parts.append(json.load(f))
    return parts


def steady_steps(part, warmup):
    return [s for s in part.get("steps", []) if s.get("epoch", 0) >= warmup]


def steady_epochs(part, warmup):
    return [e for e in part.get("epochs", []) if e.get("epoch", 0) >= warmup]


def median_positive_deltas(cum):
    d = np.diff(np.asarray(cum, dtype=float))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


# ---------------------------------------------------------- per-run stats --
def run_stats(run_dir, warmup):
    """Aggregate one run (all parts) into the quantities the fits need."""
    parts = load_parts(run_dir)
    if not parts:
        return None
    st = {"n_parts": len(parts)}
    step_t, fetch_t, hit = [], [], []
    reb_t, reb_u, reb_reuse, reb_rows, reb_acc, reb_wlen, reb_bulk = [], [], [], [], [], [], []
    own_rtt = {}      # pid -> list[(mean_rows, mean_rtt, mean_rtt_per_row)]
    cpu_w, gpu_w, ep_t = [], [], []
    bpe = []
    for part in parts:
        ss = steady_steps(part, warmup)
        if ss:
            step_t += [s["step_time_s"] for s in ss]
            fetch_t += [s["fetch_time_s"] for s in ss]
            hit += [s.get("cache_hit_pct", np.nan) / 100.0 for s in ss]
        eps = part.get("epochs", [])
        counts = {}
        for s in part.get("steps", []):
            counts[s["epoch"]] = counts.get(s["epoch"], 0) + 1
        if counts:
            bpe.append(int(np.median(list(counts.values()))))
        for r in part.get("rebuilds", []):
            if r.get("win_len", 0) <= 0:
                continue
            reb_t.append(r["t_plan_s"] + r["t_fetch_s"])
            reb_u.append(r["unique_remote_nodes"])
            reb_reuse.append(r["rows_reused"])
            reb_rows.append(r["rows_fetched"])
            reb_acc.append(r["total_remote_accesses"] / max(1, r["win_len"]))
            reb_wlen.append(r["win_len"])
            if r["rows_fetched"] > 0:
                reb_bulk.append(r["t_fetch_s"] / r["rows_fetched"])
        for e in steady_epochs(part, warmup):
            ol = e.get("owner_latency") or {}
            for pid, v in ol.items():
                if not v or v.get("n", 0) == 0:
                    continue
                own_rtt.setdefault(int(pid), []).append(
                    (v["rows"] / max(1, v["n"]), v["mean_rtt"],
                     v.get("mean_rtt_per_row", np.nan)))
        if len(eps) > warmup + 1:
            se = [e for e in eps]
            cpu = median_positive_deltas([e["cpu_energy_j"] for e in se[warmup:]])
            gpu = median_positive_deltas([e["gpu_energy_j"] for e in se[warmup:]])
            t = float(np.median([e["epoch_time_s"] for e in se[warmup:]]))
            if t > 0:
                ep_t.append(t)
                if cpu > 0:
                    cpu_w.append(cpu / t)
                if gpu > 0:
                    gpu_w.append(gpu / t)
    st["step_time"] = float(np.mean(step_t)) if step_t else np.nan
    st["fetch_time"] = float(np.mean(fetch_t)) if fetch_t else np.nan
    st["hit"] = float(np.nanmean(hit)) if hit else np.nan
    st["t_rebuild"] = float(np.median(reb_t)) if reb_t else np.nan
    st["u_nodes"] = float(np.median(reb_u)) if reb_u else np.nan
    st["reuse_frac"] = (float(np.sum(reb_reuse)) / max(1.0, float(np.sum(reb_u)))
                        if reb_u else np.nan)
    st["acc_per_batch"] = float(np.median(reb_acc)) if reb_acc else np.nan
    st["win_len"] = float(np.median(reb_wlen)) if reb_wlen else np.nan
    st["t_bulk_row"] = float(np.median(reb_bulk)) if reb_bulk else np.nan
    st["owner_rtt"] = {p: (float(np.mean([x[0] for x in v])),
                           float(np.mean([x[1] for x in v])),
                           float(np.nanmean([x[2] for x in v])))
                       for p, v in own_rtt.items()}
    st["p_cpu_w"] = float(np.median(cpu_w)) if cpu_w else np.nan
    st["p_gpu_w"] = float(np.median(gpu_w)) if gpu_w else np.nan
    st["epoch_time"] = float(np.median(ep_t)) if ep_t else np.nan
    st["bpe"] = int(np.median(bpe)) if bpe else 0
    return st


# --------------------------------------------------------------- the fits --
def fit_logistic_h(Ws, hs, boot=200, seed=0):
    Ws, hs = np.asarray(Ws, float), np.asarray(hs, float)

    def f(W, hmin, hmax, w_half, g):
        return hmin + (hmax - hmin) / (1.0 + (W / w_half) ** g)

    p0 = [max(0.01, hs.min()), min(0.99, hs.max()), np.median(Ws), 1.0]
    bounds = ([0, 0, 0.5, 0.1], [1, 1, 512, 8])
    popt, _ = curve_fit(f, Ws, hs, p0=p0, bounds=bounds, maxfev=20000)
    pred = f(Ws, *popt)
    ssr = np.sum((hs - pred) ** 2)
    sst = np.sum((hs - hs.mean()) ** 2)
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        idx = rng.integers(0, len(Ws), len(Ws))
        try:
            p, _ = curve_fit(f, Ws[idx], hs[idx], p0=popt, bounds=bounds,
                             maxfev=20000)
            bs.append(p)
        except Exception:
            pass
    ci = (np.percentile(bs, [2.5, 97.5], axis=0).tolist() if bs else None)
    return dict(hmin=popt[0], hmax=popt[1], w_half=popt[2], gamma_h=popt[3],
                r2=r2, ci95=ci), f, popt


def fit_rebuild(Ws, Ts):
    Ws, Ts = np.asarray(Ws, float), np.asarray(Ts, float)

    def f(W, a, b, c):
        return a + b * W ** c

    p0 = [max(0.0, Ts.min() * 0.5), 0.01, 0.7]
    bounds = ([0, 0, 0.05], [max(1e-9, Ts.min() * 2 + 1e-6), 10, 1.0])
    popt, _ = curve_fit(f, Ws, Ts, p0=p0, bounds=bounds, maxfev=20000)
    pred = f(Ws, *popt)
    sst = np.sum((Ts - Ts.mean()) ** 2)
    r2 = 1 - np.sum((Ts - pred) ** 2) / sst if sst > 0 else float("nan")
    return dict(a=popt[0], b=popt[1], c=popt[2], r2=r2), f, popt


def fit_footprint(Ws, Us):
    Ws, Us = np.asarray(Ws, float), np.asarray(Us, float)
    A = np.vstack([np.ones_like(Ws), np.log(Ws)]).T
    sol, *_ = np.linalg.lstsq(A, np.log(Us), rcond=None)
    return dict(U0=float(np.exp(sol[0])), u_exp=float(sol[1]))


def fit_owner_linear(owner_rtt):
    """OLS mean_rtt = t_init + t_pay*rows across owners (clean)."""
    pts = [(rows, rtt) for rows, rtt, _ in owner_rtt.values()
           if rows > 0 and np.isfinite(rtt)]
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    A = np.vstack([np.ones_like(x), x]).T
    (t_init, t_pay), *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(t_init_row=float(max(t_init, 0.0)),
                t_pay_row=float(max(t_pay, 1e-9)), n_points=len(pts))


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--victim_pid", type=int, default=3)
    ap.add_argument("--feat_dim", type=int, default=0,
                    help="features per node (0 = infer from bytes/rows/4)")
    ap.add_argument("--verification", default=None,
                    help="verification_c1.json from verify_congestion.py "
                         "(netbench ground-truth kappa) for cross-check")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if curve_fit is None:
        sys.exit("scipy required (pip install scipy)")

    ds, cd = args.dataset, args.calib_dir
    lines = [f"# Calibration report — {ds}", ""]

    # ---- Phase 1: clean W-sweep -------------------------------------------
    sweep = {}
    for d in sorted(glob.glob(os.path.join(cd, f"{ds}_clean_W*"))):
        W = int(d.rsplit("W", 1)[-1])
        st = run_stats(d, args.warmup)
        if st and np.isfinite(st["step_time"]):
            sweep[W] = st
    if len(sweep) < 4:
        sys.exit(f"need >=4 clean W runs, found {sorted(sweep)}")
    Ws = sorted(sweep)
    lines.append(f"Clean W-sweep points: {Ws}")

    hfit, hf, hpopt = fit_logistic_h(Ws, [sweep[w]["hit"] for w in Ws])
    rfit, rf, rpopt = fit_rebuild(Ws, [sweep[w]["t_rebuild"] for w in Ws])
    ufit = fit_footprint(Ws, [sweep[w]["u_nodes"] for w in Ws])
    lines += [f"h(W) fit: {hfit}", f"T_rebuild fit: {rfit}", f"U(W) fit: {ufit}"]

    # holdout for h and rebuild
    fit_on = [w for w in Ws if w in (1, 4, 16, 64)]
    held = [w for w in Ws if w not in fit_on]
    if len(fit_on) >= 4 and held:
        _, hfh, ph = fit_logistic_h(fit_on, [sweep[w]["hit"] for w in fit_on])
        errs = {w: float(abs(hfh(w, *ph) - sweep[w]["hit"])) for w in held}
        lines.append(f"h(W) holdout abs err (fit {fit_on} -> test): {errs}")

    t_base_cands, t_bulk = [], []
    for w in Ws:
        s = sweep[w]
        amort = rf(w, *rpopt) / max(1.0, w)
        t_base_cands.append(s["step_time"] - amort - max(0.0, s["fetch_time"]))
        if np.isfinite(s["t_bulk_row"]):
            t_bulk.append(s["t_bulk_row"])
    t_base = float(max(1e-4, np.median(sorted(t_base_cands)[:3])))
    t_bulk_row = float(np.median(t_bulk)) if t_bulk else 3.5e-7

    ow = fit_owner_linear(sweep[max(Ws)]["owner_rtt"]) or {}
    R = float(np.nanmedian([sweep[w]["acc_per_batch"] for w in Ws]))
    reuse = float(np.nanmedian([sweep[w]["reuse_frac"] for w in Ws]))
    bpe = int(np.median([sweep[w]["bpe"] for w in Ws if sweep[w]["bpe"]]))
    p_cpu = float(np.nanmedian([sweep[w]["p_cpu_w"] for w in Ws]))
    p_gpu = float(np.nanmedian([sweep[w]["p_gpu_w"] for w in Ws]))
    lines += [f"t_base={t_base:.5f}s t_bulk_row={t_bulk_row:.3e}s "
              f"R_per_batch={R:.1f} reuse_frac={reuse:.3f} bpe={bpe}",
              f"owner linear fit: {ow}",
              f"power: p_cpu_w={p_cpu:.1f} p_gpu_w={p_gpu:.1f}"]

    # ---- Phase 2: congestion axes -----------------------------------------
    clean16 = sweep.get(16)
    kappa_tbl, sigma_tbl = {}, {}
    for d in sorted(glob.glob(os.path.join(cd, f"{ds}_c1_*_W16"))):
        rate = d.split("_c1_")[-1].replace("_W16", "")
        st = run_stats(d, args.warmup)
        if not st or not clean16:
            continue
        v = st["owner_rtt"].get(args.victim_pid)
        c = clean16["owner_rtt"].get(args.victim_pid)
        others = [st["owner_rtt"][p][2] / clean16["owner_rtt"][p][2]
                  for p in st["owner_rtt"]
                  if p != args.victim_pid and p in clean16["owner_rtt"]
                  and np.isfinite(st["owner_rtt"][p][2])
                  and clean16["owner_rtt"][p][2] > 0]
        if v and c and c[2] > 0 and np.isfinite(v[2]):
            kappa_tbl[rate] = {
                "kappa_inband": round(v[2] / c[2], 2),
                "others_ratio": round(float(np.mean(others)), 2) if others else None,
                "step_time_ratio": round(st["step_time"] / clean16["step_time"], 3),
            }
    for d in sorted(glob.glob(os.path.join(cd, f"{ds}_c4_*_W16"))):
        ms = d.split("_c4_")[-1].replace("ms_W16", "")
        st = run_stats(d, args.warmup)
        if not st or not clean16:
            continue
        v = st["owner_rtt"].get(args.victim_pid)
        c = clean16["owner_rtt"].get(args.victim_pid)
        if v and c:
            sigma_tbl[ms] = {
                "rtt_delta_ms": round((v[1] - c[1]) * 1e3, 2),
                "step_time_ratio": round(st["step_time"] / clean16["step_time"], 3),
                "fetch_time_ratio": round(st["fetch_time"] /
                                          max(1e-9, clean16["fetch_time"]), 3),
            }
    lines += [f"kappa table (c1, in-band): {json.dumps(kappa_tbl, indent=1)}",
              f"sigma table (c4): {json.dumps(sigma_tbl, indent=1)}"]

    if args.verification and os.path.exists(args.verification):
        with open(args.verification) as f:
            ver = json.load(f)
        lines.append(f"netbench ground-truth cross-check: "
                     f"{json.dumps(ver.get('severities', ver), indent=1)[:2000]}")

    # c_ar: residual step inflation under delay beyond fetch inflation
    c_ar = 0.002
    if sigma_tbl:
        resid = [(v["step_time_ratio"] - 1) - (v["fetch_time_ratio"] - 1) *
                 (clean16["fetch_time"] / clean16["step_time"])
                 for v in sigma_tbl.values()]
        c_ar = float(max(0.0, np.median(resid)) * clean16["step_time"])
        lines.append(f"c_ar (AllReduce straggler residual) = {c_ar:.5f}s per unit sigma")

    feat_dim = args.feat_dim
    if not feat_dim:
        # infer from any rebuild bytes/rows
        for w in Ws:
            p = load_parts(os.path.join(cd, f"{ds}_clean_W{w}"))
            for part in p:
                for r in part.get("rebuilds", []):
                    if r.get("rows_fetched", 0) > 0:
                        feat_dim = round(r["bytes_fetched"] / r["rows_fetched"] / 4)
                        break
                if feat_dim: break
            if feat_dim: break

    calib = {
        "_provenance": {"dataset": ds, "calib_dir": os.path.abspath(cd),
                        "warmup": args.warmup, "generated_by": "fit_calibration.py",
                        "clean_Ws": Ws, "h_r2": round(hfit["r2"], 4),
                        "rebuild_r2": round(rfit["r2"], 4),
                        "kappa_table": kappa_tbl, "sigma_table": sigma_tbl},
        "t_init_row": ow.get("t_init_row", 6.0e-05),
        "t_pay_row": ow.get("t_pay_row", 5.7e-06),
        "rpc_rows": 64,
        "q_depth": 4,
        "a": round(rfit["a"], 6), "b": round(rfit["b"], 6), "c": round(rfit["c"], 4),
        "t_bulk_row": t_bulk_row,
        "alpha_crit": 0.35,
        "hmin": round(hfit["hmin"], 4), "hmax": round(hfit["hmax"], 4),
        "w_half": round(hfit["w_half"], 3), "gamma_h": round(hfit["gamma_h"], 4),
        "t_base": round(t_base, 6),
        "feat_bytes": int(feat_dim * 4) if feat_dim else 2408,
        "R_per_batch": round(R, 1),
        "batches_per_epoch": bpe or 100,
        "n_epochs": 30,
        "n_hot": 100000,
        "P": 4,
        "U0": round(ufit["U0"], 1), "u_exp": round(ufit["u_exp"], 4),
        "reuse_frac": round(reuse, 4),
        "p_cpu_w": round(p_cpu, 1) if np.isfinite(p_cpu) else 180.0,
        "p_gpu_active_w": round(p_gpu, 1) if np.isfinite(p_gpu) else 120.0,
        "p_gpu_idle_w": 30.0,
        "gpu_active_frac": 0.6,
        "c_ar": round(c_ar, 6),
        "alloc_tilt": 0.8,
        "alpha_rpc": 0.00467, "beta": 1.4e-9, "gamma_c": 2.01e-10,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=1)
    print(f"wrote {args.out}")

    rep = args.report or os.path.join(cd, f"calib_report_{ds}.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {rep}")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
