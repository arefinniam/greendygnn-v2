#!/usr/bin/env python3
"""Collect per-batch remote-access traces (RESEARCH_PLAN_v2 item 3, Layer 1).

Runs the PRODUCTION sampling path (same DistSampler, same seeded shuffle, same
node_split) with no training, and dumps every batch's remote node IDs +
owners. A trace collected with seed s is therefore byte-identical to the
access stream a training run with seed s consumes — which is what makes the
Layer-1 cache engine trace-EXACT rather than trace-approximate.

From the trace, h(W,C,lambda), U(W), rebuild rows/bytes, per-owner miss
composition, and cache overlap across any candidate schedule are computable
offline for ANY (W, C, allocation) — no cluster time.

Launch exactly like a trainer (one rank per partition, via launch.py or the
matrix driver's mechanism):
  python3 collect_trace.py --graph_name reddit --ip_config ... --part_config \
      ... --batch_size 2000 --fan_out 10,25 --num_epochs 30 --seed 0 --out_dir ...

Output per rank: {out_dir}/trace_part{pid}.pt
  {"meta": {...}, "batches": [{"remote": int32[K], "owners": uint8[K],
                               "n_inputs": int, "n_seeds": int}, ...]}
(batches are in consumption order across all epochs; epoch boundaries at
multiples of meta["bpe"]).
"""

import argparse
import hashlib
import os
import time

import torch as th
import dgl
import dgl.distributed

from helpers import set_all_seeds
from sampler import DistSampler


def main(args):
    gen = set_all_seeds(args.seed)
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    train_nid = dgl.distributed.node_split(
        g.ndata["train_mask"], g.get_partition_book(), force_even=True)
    pb = g.get_partition_book()
    pid = g.rank()

    lp = g.local_partition
    if lp:
        inner = lp.ndata["inner_node"].bool()
        ids = lp.ndata["_ID"]
        if not isinstance(ids, th.Tensor):
            ids = th.tensor(ids, dtype=th.long)
        lid = ids[inner]
    else:
        lid = th.empty(0, dtype=th.long)
    lmask = th.zeros(g.num_nodes(), dtype=th.bool)
    if lid.numel() > 0:
        lmask[lid] = True

    sampler = DistSampler(g, train_nid, args.fan_out, args.batch_size,
                          generator=gen)
    bpe = len(sampler)
    print(f"Part {pid}: tracing {args.graph_name} bpe={bpe} "
          f"epochs={args.num_epochs} seed={args.seed}")

    batches = []
    t0 = time.time()
    for epoch in range(args.num_epochs):
        for input_nodes, seeds, _blocks in sampler:
            rmask = ~lmask[input_nodes]
            remote = input_nodes[rmask]
            owners = pb.nid2partid(remote)
            batches.append({
                "remote": remote.to(th.int32),
                "owners": owners.to(th.uint8),
                "n_inputs": int(input_nodes.numel()),
                "n_seeds": int(seeds.numel()),
            })
        print(f"Part {pid}: epoch {epoch} traced "
              f"({len(batches)} batches, {time.time() - t0:.0f}s)")

    with open(args.part_config, "rb") as f:
        part_md5 = hashlib.md5(f.read()).hexdigest()
    meta = dict(
        dataset=args.graph_name, part_config=args.part_config,
        part_config_md5=part_md5, num_nodes=int(g.num_nodes()),
        P=int(pb.num_partitions()), pid=int(pid),
        feat_dim=int(g.ndata["features"].shape[1]),
        feat_dtype=str(g.ndata["features"].dtype),
        batch_size=args.batch_size, fan_out=args.fan_out,
        num_epochs=args.num_epochs, bpe=bpe, seed=args.seed,
        n_train_local=int(train_nid.numel()), collected_t=time.time(),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"trace_part{pid}.pt")
    th.save({"meta": meta, "batches": batches}, out)
    sz = os.path.getsize(out) / 1e6
    print(f"Part {pid}: wrote {out} ({len(batches)} batches, {sz:.0f} MB)")
    th.distributed.barrier()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, t, d in [
        ("--graph_name", str, None), ("--ip_config", str, None),
        ("--part_config", str, None), ("--backend", str, "gloo"),
        ("--num_epochs", int, 30), ("--fan_out", str, "10,25"),
        ("--batch_size", int, 2000), ("--seed", int, 0),
        ("--out_dir", str, "traces"), ("--local_rank", int, None),
        ("--num_gpus", int, 0), ("--n_classes", int, 0),
    ]:
        p.add_argument(a, type=t, default=d)
    main(p.parse_args())
