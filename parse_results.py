#!/usr/bin/env python3
"""Results parser v2 — from raw run artifacts to the paper table, no hands.

Consumes, per run directory (as laid out by run_matrix.sh:
  <matrix_dir>/runs/<dataset>__<condition>__<method>__seed<k>/):
  * run.log                       unified SUMMARY lines from all trainers:
                                    SUMMARY method=<m> part=<r> gpu_j=<x> ...
  * <label>_part<p>_profile.json  TrainingProfiler output (steps[]/epochs[],
                                  incl. I5 fields when present: remote_bytes,
                                  remote_rows, remote_fetch_count, khat, ...)
  * p1_aggregate.json             wrap-robust cluster aggregate (aggregate_p1)
  * cong_journal.jsonl            congestion transition journal (optional)
  * run_meta.json                 {dataset, condition, method, seed, t_launch,
                                   t_end, exit_code, ...} written by the driver

Emits results_table.json: per (dataset, condition, method) mean +/- std over
seeds for EVERY numeric metric found, with full provenance (run dirs, seeds,
journals). Optionally a markdown summary table. There are no hand-entered
numbers anywhere in this pipeline: figures/tables must be regenerated from
this file only.

Usage:
  python3 parse_results.py --matrix_dir results_matrix --out results_table.json \
      [--md results_table.md]
  python3 parse_results.py --log_file run.log --output metrics.json   # single run
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Primitive parsers (pure; unit-tested)
# ---------------------------------------------------------------------------

SUMMARY_RE = re.compile(r"^SUMMARY\s+(.*)$")
KV_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_summary_lines(text):
    """Parse unified 'SUMMARY k=v k=v ...' lines -> list of dicts.
    Values parse as float when possible, else stay strings."""
    out = []
    for line in text.splitlines():
        m = SUMMARY_RE.match(line.strip())
        if not m:
            continue
        rec = {}
        for k, v in KV_RE.findall(m.group(1)):
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
        if rec:
            out.append(rec)
    return out


def flatten_numeric(obj, prefix="", depth=2):
    """Flatten nested dict -> {dotted_key: float} for numeric leaves."""
    flat = {}
    if depth < 0:
        return flat
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                flat[key] = float(v)
            elif isinstance(v, dict):
                flat.update(flatten_numeric(v, key, depth - 1))
    return flat


def mean_std(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mu = sum(vals) / n
    sd = (sum((v - mu) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd, n


def profile_derived_metrics(profile):
    """Per-run metrics derived from one TrainingProfiler JSON (any part)."""
    steps = profile.get("steps", [])
    epochs = profile.get("epochs", [])
    d = {}
    for field, out_name in [("remote_bytes", "remote_bytes_total"),
                            ("remote_rows", "remote_rows_total"),
                            ("remote_fetch_count", "remote_fetches_total")]:
        vals = [s.get(field) for s in steps if s.get(field) is not None]
        if vals:
            d[out_name] = float(sum(vals))
    hits = [s.get("cache_hit_pct") for s in steps
            if s.get("cache_hit_pct") is not None]
    if hits:
        d["cache_hit_pct_mean"] = sum(hits) / len(hits)
    accs = [e.get("avg_accuracy") for e in epochs
            if e.get("avg_accuracy") is not None]
    if accs:
        d["final_accuracy"] = accs[-1]
        d["best_accuracy"] = max(accs)
    ov = [s.get("controller_overhead_ms") for s in steps
          if s.get("controller_overhead_ms") is not None]
    ov += [e.get("controller_overhead_ms") for e in epochs
           if e.get("controller_overhead_ms") is not None]
    if ov:
        d["controller_overhead_ms_mean"] = sum(ov) / len(ov)
    times = [e.get("epoch_time_s") for e in epochs
             if e.get("epoch_time_s") is not None]
    if times:
        stable = times[max(0, len(times) - int(len(times) * 0.8)):]
        d["epoch_time_s_stable_mean"] = sum(stable) / len(stable)
    return d


def load_journal(path):
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def journal_on_windows(events):
    """Reconstruct absolute [t_on, t_off] impairment windows from a journal.
    ON at 'applied'/'wave_on'/'phase_on'; OFF at 'removed'/'wave_off'/
    'phase_off'/'teardown_verified'/'run_end'."""
    on_actions = {"applied", "wave_on", "phase_on"}
    off_actions = {"removed", "wave_off", "phase_off", "teardown_verified",
                   "run_end"}
    windows, t_on = [], None
    for ev in sorted(events, key=lambda e: e.get("t", 0)):
        a = ev.get("action")
        if a in on_actions and t_on is None:
            t_on = ev["t"]
        elif a in off_actions and t_on is not None:
            windows.append((t_on, ev["t"]))
            t_on = None
    if t_on is not None:
        windows.append((t_on, t_on))  # never closed: zero-length, flagged
    return windows


def realized_exposure(windows, t0, t1):
    if t1 <= t0:
        return 0.0
    cov = sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in windows)
    return cov / (t1 - t0)


def per_epoch_exposure(windows, t_end_run, part0_profile):
    """Approximate per-epoch exposure. The profiler's clock is relative to
    trainer start; we anchor it as t_end_run - total_wall_time_s (save()
    runs at trainer exit; residual error = post-save teardown seconds,
    negligible against >=60 s wave periods). Documented approximation."""
    total_wall = part0_profile.get("total_wall_time_s")
    epochs = part0_profile.get("epochs", [])
    if not windows or total_wall is None or not epochs:
        return None
    t_start = t_end_run - total_wall
    out, prev = [], 0.0
    for e in epochs:
        w = e.get("wall_time_s")
        if w is None:
            return None
        out.append({"epoch": e.get("epoch"),
                    "exposure": round(realized_exposure(
                        windows, t_start + prev, t_start + w), 3)})
        prev = w
    return out


# ---------------------------------------------------------------------------
# Run / matrix scanning
# ---------------------------------------------------------------------------

RUN_DIR_RE = re.compile(
    r"^(?P<dataset>[^_].*?)__(?P<condition>.+?)__(?P<method>.+?)__seed(?P<seed>\d+)$")


def parse_run_dir(run_dir):
    """Extract everything measurable from one run directory."""
    run_dir = Path(run_dir)
    rec = {"run_dir": str(run_dir), "metrics": {}, "provenance": {}}

    meta_p = run_dir / "run_meta.json"
    meta = {}
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        rec["meta"] = meta

    agg_p = run_dir / "p1_aggregate.json"
    if agg_p.exists():
        rec["metrics"].update(
            {f"agg.{k}": v for k, v in
             flatten_numeric(json.loads(agg_p.read_text())).items()})
        rec["provenance"]["aggregate"] = str(agg_p)

    profiles = sorted(run_dir.glob("*_part*_profile.json"))
    rec["provenance"]["profiles"] = [str(p) for p in profiles]
    part0 = None
    sum_totals = {}
    for p in profiles:
        try:
            prof = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if prof.get("part_id") == 0:
            part0 = prof
        for k, v in profile_derived_metrics(prof).items():
            if k in ("remote_bytes_total", "remote_rows_total",
                     "remote_fetches_total"):
                sum_totals[k] = sum_totals.get(k, 0.0) + v
            elif prof.get("part_id") == 0:
                rec["metrics"][f"prof.{k}"] = v
    for k, v in sum_totals.items():
        rec["metrics"][f"prof.{k}_allparts"] = v
    rec["n_profiles"] = len(profiles)

    log_p = run_dir / "run.log"
    if log_p.exists():
        summaries = parse_summary_lines(log_p.read_text(errors="replace"))
        rec["provenance"]["run_log"] = str(log_p)
        by_part = {}
        for s in summaries:
            part = int(s.get("part", -1))
            by_part[part] = s
        for agg_key in ("gpu_j", "cpu_j", "total_j"):
            vals = [s[agg_key] for s in by_part.values()
                    if isinstance(s.get(agg_key), float)]
            if vals:
                rec["metrics"][f"summary.{agg_key}_sum_allparts"] = sum(vals)

    j_p = run_dir / "cong_journal.jsonl"
    if j_p.exists():
        events = load_journal(j_p)
        windows = journal_on_windows(events)
        rec["provenance"]["journal"] = str(j_p)
        t0 = meta.get("t_launch")
        t1 = meta.get("t_end")
        if t0 and t1:
            rec["metrics"]["exposure.run_fraction"] = round(
                realized_exposure(windows, t0, t1), 4)
            if part0 is not None:
                pe = per_epoch_exposure(windows, t1, part0)
                if pe:
                    rec["per_epoch_exposure"] = pe
        rec["metrics"]["exposure.n_transitions"] = float(len(windows))
    return rec


def scan_matrix(matrix_dir):
    runs_root = Path(matrix_dir) / "runs"
    if not runs_root.exists():
        runs_root = Path(matrix_dir)
    groups = {}
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_DIR_RE.match(d.name)
        if not m:
            continue
        key = (m["dataset"], m["condition"], m["method"])
        rec = parse_run_dir(d)
        rec["seed"] = int(m["seed"])
        groups.setdefault(key, []).append(rec)
    return groups


def aggregate_groups(groups):
    table = []
    for (dataset, condition, method), runs in sorted(groups.items()):
        all_keys = sorted({k for r in runs for k in r["metrics"]})
        agg = {}
        for k in all_keys:
            mu, sd, n = mean_std([r["metrics"].get(k) for r in runs])
            if mu is not None:
                agg[k] = {"mean": mu, "std": sd, "n": n}
        table.append({
            "dataset": dataset, "condition": condition, "method": method,
            "n_seeds": len(runs),
            "seeds": sorted(r["seed"] for r in runs),
            "metrics": agg,
            "provenance": [{"run_dir": r["run_dir"], "seed": r["seed"],
                            "n_profiles": r.get("n_profiles", 0),
                            **r["provenance"]} for r in runs],
        })
    return table


MD_CANDIDATES = [
    ("agg.total_j_per_epoch", "J/ep"),
    ("agg.sys_total_j_per_epoch", "J/ep"),
    ("agg.cpu_j_per_epoch", "cpuJ/ep"),
    ("agg.sys_cpu_j_per_epoch", "cpuJ/ep"),
    ("agg.epoch_time_s", "ep_t(s)"),
    ("prof.epoch_time_s_stable_mean", "ep_t(s)"),
    ("prof.final_accuracy", "acc"),
    ("agg.accuracy", "acc"),
    ("prof.cache_hit_pct_mean", "hit%"),
    ("prof.remote_bytes_total_allparts", "rem_MB"),
    ("exposure.run_fraction", "expo"),
]


def to_markdown(table):
    cols, seen = [], set()
    for row in table:
        for key, label in MD_CANDIDATES:
            if key in row["metrics"] and label not in seen:
                cols.append((key, label))
                seen.add(label)
    header = ["dataset", "condition", "method", "seeds"] + [l for _, l in cols]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in table:
        cells = [row["dataset"], row["condition"], row["method"],
                 str(row["n_seeds"])]
        for key, label in cols:
            m = row["metrics"].get(key)
            if m is None:
                cells.append("-")
            else:
                v, s = m["mean"], m["std"]
                if label == "rem_MB":
                    cells.append(f"{v / 1e6:.0f}±{s / 1e6:.0f}")
                else:
                    cells.append(f"{v:.3g}±{s:.2g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Single-run (legacy-ish) mode
# ---------------------------------------------------------------------------

def single_run(log_file, output, method, dataset, batch_size):
    text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
    summaries = parse_summary_lines(text)
    run_dir = str(Path(log_file).parent)
    rec = parse_run_dir(run_dir)
    out = {"meta": {"method": method, "dataset": dataset,
                    "batch_size": batch_size, "log_file": log_file},
           "summaries": summaries,
           "metrics": rec["metrics"],
           "provenance": rec["provenance"]}
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    tj = rec["metrics"].get("summary.total_j_sum_allparts")
    print(f"[{method}] {dataset} B={batch_size}: "
          f"summaries={len(summaries)} total_j={tj}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix_dir", help="matrix results root (matrix mode)")
    ap.add_argument("--out", default="results_table.json")
    ap.add_argument("--md", default="", help="also write a markdown table")
    ap.add_argument("--log_file", help="single-run mode: one run.log")
    ap.add_argument("--output", default="metrics.json")
    ap.add_argument("--method", default="unknown")
    ap.add_argument("--dataset", default="unknown")
    ap.add_argument("--batch_size", default="unknown")
    args = ap.parse_args()

    if args.matrix_dir:
        groups = scan_matrix(args.matrix_dir)
        table = aggregate_groups(groups)
        payload = {"generated_by": "parse_results.py v2",
                   "matrix_dir": os.path.abspath(args.matrix_dir),
                   "n_configs": len(table),
                   "table": table}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"-> {args.out} ({len(table)} configs)")
        if args.md:
            with open(args.md, "w") as f:
                f.write(to_markdown(table))
            print(f"-> {args.md}")
    elif args.log_file:
        single_run(args.log_file, args.output, args.method, args.dataset,
                   args.batch_size)
    else:
        ap.error("need --matrix_dir or --log_file")
        sys.exit(2)


if __name__ == "__main__":
    main()
