#!/usr/bin/env python3
"""Aggregate a Priority-1 run into a clean steady-state energy/metric row.

Collects the 4 per-part *_profile.json (part0 local on gnn1, parts 1-3 pulled
from peers over the private net), then derives per-epoch energy from the
CUMULATIVE per-epoch energy via first differences, discarding `--warmup`
epochs so DGL startup / barrier-wait energy is excluded from the steady state.

Run ON gnn1 (it can reach peers at 10.52.x).
"""
import argparse, glob, json, os, subprocess, statistics, sys

PEERS = {1: "10.52.3.217", 2: "10.52.3.123", 3: "10.52.3.89"}  # part_id -> private IP


def collect(run_dir, label):
    """Ensure all 4 part profiles are present locally under run_dir.

    Peer scp can fail transiently right after a run (peer busy / under netem), so
    retry a few times. Warn LOUDLY if fewer than 4 parts are collected — a partial
    collection silently under-counts system energy (the het_point bug)."""
    found = {}
    for pid in range(4):
        local = os.path.join(run_dir, f"{label}_part{pid}_profile.json")
        if pid in PEERS and not os.path.exists(local):
            for _ in range(4):
                subprocess.run(["scp", "-o", "StrictHostKeyChecking=no",
                                "-o", "ConnectTimeout=8",
                                f"cc@{PEERS[pid]}:{local}", local],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(local):
                    break
        if os.path.exists(local):
            found[pid] = json.load(open(local))
    if len(found) < 4:
        missing = [p for p in range(4) if p not in found]
        print(f"WARNING: only {len(found)}/4 part profiles collected for "
              f"'{label}' (missing parts {missing}) — system energy UNDER-COUNTS.",
              file=sys.stderr)
    return found


def _robust(deltas):
    """(median, mean, dropped) of physical (positive) per-epoch deltas.

    RAPL is a free-running counter that wraps at max_energy_range_uj (~262 kJ)
    across a sweep; a wrap injects one large-negative delta, so both estimators
    use positive deltas only (wrap-immune).

    Median vs mean: under DUTY-CYCLED congestion (c2/c3) per-epoch energy is
    bimodal and the median selects the majority phase (typically the
    uncongested one), under-weighting congested and rebuild-spike epochs; the
    mean is the time-weighted average. Report BOTH; small-margin conclusions
    must hold under both (2026-07-10 audit)."""
    pos = [d for d in deltas if d > 0]
    dropped = len(deltas) - len(pos)
    if not pos:
        return float('nan'), float('nan'), dropped
    return statistics.median(pos), statistics.mean(pos), dropped


def per_epoch_energy(epochs, warmup):
    """Cumulative per-epoch energy -> wrap-robust steady-state per-epoch energy."""
    if len(epochs) < warmup + 2:
        return None
    e = sorted(epochs, key=lambda x: x["epoch"])
    cpu = [x.get("cpu_energy_j", 0.0) for x in e]
    gpu = [x.get("gpu_energy_j", 0.0) for x in e]
    t   = [x.get("epoch_time_s", 0.0) for x in e]
    # epoch i's delta corresponds to epochs[i]; keep i-1 >= warmup (steady state)
    idx = [k for k in range(1, len(cpu)) if k >= warmup]
    if not idx:
        return None
    dcpu = [cpu[k] - cpu[k-1] for k in idx]
    dgpu = [gpu[k] - gpu[k-1] for k in idx]
    mc, ac, dc = _robust(dcpu)
    mg, ag, dg = _robust(dgpu)
    mt = statistics.median([t[k] for k in idx])
    return dict(cpu_j=mc, gpu_j=mg, total_j=mc + mg,
                cpu_j_mean=ac, gpu_j_mean=ag, total_j_mean=ac + ag,
                epoch_time_s=mt,
                n_ss=len(idx), cpu_wraps=dc, gpu_wraps=dg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--label", required=True, help="e.g. optisched_static")
    ap.add_argument("--warmup", type=int, default=3, help="epochs to discard")
    ap.add_argument("--batch_size", type=int, default=2000)
    args = ap.parse_args()

    parts = collect(args.run_dir, args.label)
    if not parts:
        print("no profiles found", file=sys.stderr); sys.exit(1)

    rows, sys_cpu, sys_gpu, times, accs, hits = {}, 0.0, 0.0, [], [], []
    sys_cpu_mean, sys_gpu_mean = 0.0, 0.0
    for pid, d in sorted(parts.items()):
        pe = per_epoch_energy(d.get("epochs", []), args.warmup)
        steps = d.get("steps", [])
        hr = [s["cache_hit_pct"] for s in steps if s.get("cache_hit_pct") is not None]
        acc = d["epochs"][-1].get("avg_accuracy") if d.get("epochs") else None
        bpe = max((s.get("step", 0) for s in steps), default=0) + 1
        rows[pid] = dict(pe=pe, bpe=bpe, hit=(statistics.mean(hr) if hr else None), acc=acc)
        if pe:
            sys_cpu += pe["cpu_j"]; sys_gpu += pe["gpu_j"]; times.append(pe["epoch_time_s"])
            sys_cpu_mean += pe["cpu_j_mean"]; sys_gpu_mean += pe["gpu_j_mean"]
        if acc is not None: accs.append(acc)
        if hr: hits.append(statistics.mean(hr))

    print(f"\n== Priority-1 steady-state row  ({args.label}, warmup={args.warmup}) ==")
    print(f"{'part':>4} {'cpu_J/ep':>9} {'gpu_J/ep':>9} {'tot_J/ep':>9} {'ep_t_s':>7} {'hit%':>6} {'acc':>6} {'n_ss':>4}")
    for pid, r in rows.items():
        pe = r["pe"] or {}
        print(f"{pid:>4} {pe.get('cpu_j',float('nan')):>9.1f} {pe.get('gpu_j',float('nan')):>9.1f} "
              f"{pe.get('total_j',float('nan')):>9.1f} {pe.get('epoch_time_s',float('nan')):>7.2f} "
              f"{(r['hit'] or float('nan')):>6.1f} {(r['acc'] or float('nan')):>6.3f} {pe.get('n_ss',0):>4}")
    bpe = max((r["bpe"] for r in rows.values()), default=0)
    ep_t = statistics.mean(times) if times else float('nan')
    thru = (len(parts) * bpe * args.batch_size) / ep_t if ep_t and ep_t == ep_t else float('nan')
    out = dict(
        label=args.label, n_parts=len(parts), warmup=args.warmup,
        sys_cpu_j_per_epoch=round(sys_cpu, 1), sys_gpu_j_per_epoch=round(sys_gpu, 1),
        sys_total_j_per_epoch=round(sys_cpu + sys_gpu, 1),
        # time-weighted (mean-of-positive-deltas) estimator alongside the
        # median: required for c2/c3 where per-epoch energy is bimodal.
        sys_cpu_j_per_epoch_mean=round(sys_cpu_mean, 1),
        sys_gpu_j_per_epoch_mean=round(sys_gpu_mean, 1),
        sys_total_j_per_epoch_mean=round(sys_cpu_mean + sys_gpu_mean, 1),
        mean_epoch_time_s=round(ep_t, 3) if ep_t == ep_t else None,
        throughput_samples_per_s=round(thru, 0) if thru == thru else None,
        mean_accuracy=round(statistics.mean(accs), 4) if accs else None,
        mean_cache_hit_pct=round(statistics.mean(hits), 1) if hits else None,
        bpe=bpe,
    )
    print("\nSYSTEM (sum over parts), per steady-state epoch:")
    for k, v in out.items():
        print(f"  {k}: {v}")
    json.dump(out, open(os.path.join(args.run_dir, "p1_aggregate.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(args.run_dir, 'p1_aggregate.json')}")


if __name__ == "__main__":
    main()
