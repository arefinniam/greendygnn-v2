#!/usr/bin/env python3
"""Offline schedule / regime-library builder (proposal §7.1-7.2, §6.5).

Consumes the dumped traces (dump_trace.py) and a calibrated cost model, and emits
one of two JSON artifacts consumed by train_optisched.py:

  --mode clairvoyant   per-epoch Theorem-1 DP schedule under a fixed regime
                       (the offline-optimal "ceiling" schedule).  JSON:
                         {"window_lengths_per_epoch": [[w...], ...]}

  --mode library       the online expert library (default): per regime, the
                       per-epoch DP schedule, plus a full-information loss tensor
                       loss[i][q][e] = cost of expert i's epoch-e schedule under
                       regime q -- everything the fixed-share controller needs.

Build is cheap: residual misses are sigma-independent, so each epoch is one
`precompute` followed by a DP per regime.
"""

import argparse
import glob
import json
import os
from typing import List

import numpy as np

from optisched.calibration import CostModel, default_model
from optisched.trace import Trace
from optisched.interval_cost import precompute, make_templates
from optisched.regime import default_regimes
from optisched import dp_solver as DP


def schedule_cost(cost_mat: np.ndarray, windows) -> float:
    tot = 0.0
    for (s, w) in windows:
        tot += float(cost_mat[s, w - 1])
    return tot


def load_epoch_traces(root: str, dataset: str, part: int) -> List[Trace]:
    files = sorted(glob.glob(os.path.join(root, dataset, f"r{part}_epoch_*.npz")))
    if not files:  # fallback to un-prefixed
        files = sorted(glob.glob(os.path.join(root, dataset, "epoch_*.npz")))
    return [Trace.load(f) for f in files]


def main(args):
    model = (CostModel.load(args.model) if args.model
             else default_model(num_partitions=args.num_partitions, w_max=args.w_max))
    traces = load_epoch_traces(args.traces, args.dataset, args.part)
    if not traces:
        raise SystemExit(f"no traces under {args.traces}/{args.dataset}")
    E = len(traces)
    P = traces[0].num_partitions
    local = traces[0].local_rank
    templates = make_templates(P, local) if args.allocation else None
    print(f"{args.dataset}: {E} epoch traces, B={traces[0].num_batches}, P={P}")

    # precompute interval tables once per epoch (sigma-independent)
    ics = [precompute(tr, model, args.n_hot, w_max=args.w_max, templates=templates)
           for tr in traces]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "clairvoyant":
        sigma = np.ones(P)
        wpe = []
        for ic in ics:
            cost, bt = ic.cost_matrix(model, sigma, variant="A")
            sched = DP.solve_optionA(cost, bt, w_max=args.w_max)
            wpe.append([w for (_, w) in sched.windows])
        payload = {"dataset": args.dataset, "mode": "clairvoyant",
                   "num_partitions": P, "window_lengths_per_epoch": wpe,
                   "model": model.to_dict()}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"clairvoyant schedule -> {args.out} "
              f"(mean windows/epoch={np.mean([len(x) for x in wpe]):.1f})")
        return

    # mode == library
    regimes = default_regimes(P, local)
    names = [n for n, _ in regimes]
    sigmas = [s for _, s in regimes]
    N = len(sigmas)

    # per regime, per epoch: DP schedule (window list) + its window tuples
    sched_windows = [[None] * E for _ in range(N)]
    schedules = [[None] * E for _ in range(N)]
    # cost matrices per (epoch, regime) for the loss tensor
    cost_cache = [[None] * N for _ in range(E)]
    for e, ic in enumerate(ics):
        for q in range(N):
            cq, btq = ic.cost_matrix(model, sigmas[q], variant="A")
            cost_cache[e][q] = cq
            sched = DP.solve_optionA(cq, btq, w_max=args.w_max)
            sched_windows[q][e] = sched.windows
            schedules[q][e] = [w for (_, w) in sched.windows]

    # full-information loss tensor: loss[i][q][e]
    loss = np.zeros((N, N, E))
    for i in range(N):
        for q in range(N):
            for e in range(E):
                loss[i, q, e] = schedule_cost(cost_cache[e][q], sched_windows[i][e])

    payload = {
        "dataset": args.dataset, "mode": "library", "num_partitions": P,
        "names": names, "sigmas": [s.tolist() for s in sigmas],
        "schedules": schedules,             # [N][E] -> [window lengths]
        "loss_tensor": loss.tolist(),       # [N][N][E]
        "model": model.to_dict(),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"library ({N} experts x {E} epochs) -> {args.out}")
    # quick diagnostic: best fixed expert vs per-epoch oracle expert
    fixed = loss.sum(axis=2)  # [i][q] summed over epochs; diag = own-regime cost
    print(f"  clean-expert idx={names.index('clean')}, "
          f"loss spread over experts (epoch0): "
          f"{np.round(loss[:, 0, 0], 4).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--traces", default="traces")
    p.add_argument("--dataset", required=True)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--model", default="")
    p.add_argument("--num_partitions", type=int, default=4)
    p.add_argument("--n_hot", type=int, default=100000)
    p.add_argument("--w_max", type=int, default=64)
    p.add_argument("--allocation", action="store_true")
    p.add_argument("--mode", choices=["library", "clairvoyant"], default="library")
    p.add_argument("--out", default="library.json")
    main(p.parse_args())
