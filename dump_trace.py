#!/usr/bin/env python3
"""Dump the deterministic per-batch remote-access trace from the DGL sampler.

This realises proposal A1 in the existing harness: with a seeded sampler the
per-epoch receptive-field sets are known before training, so we replay the
sampler once (no model, no training) and record, per batch, the remote input
nodes and their owner partitions.  Output feeds run_gate.py and build_library.py.

Layout written:  <out>/<dataset>/r<rank>_epoch_<e:03d>.npz   (one per epoch/rank)
Each worker dumps its own partition's trace (scheduling is per-worker).

Launch exactly like the trainers (via launch.py), e.g.:
  python3 dump_trace.py --graph_name reddit --ip_config ip_config.txt \
      --part_config $DATASET_ROOT/Reddit/data/reddit.json \
      --batch_size 2000 --num_epochs 5 --out traces --seed 1
"""

import argparse
import os
import threading

import numpy as np
import torch as th
import dgl
import dgl.distributed

from sampler import DistSampler
from optisched.trace import Trace


def main(args):
    # Seed everything so the trace is deterministic (A1).
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    dgl.seed(args.seed)

    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    train_nid = dgl.distributed.node_split(
        g.ndata["train_mask"], g.get_partition_book(), force_even=True)

    pid = g.rank()
    pb = g.get_partition_book()
    nparts = th.distributed.get_world_size()

    # local node mask (inner nodes of this partition)
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

    out_dir = os.path.join(args.out, args.graph_name)
    os.makedirs(out_dir, exist_ok=True)

    sampler = DistSampler(g, train_nid, args.fan_out, args.batch_size)
    dist_lock = threading.Lock()
    print(f"Part {pid}: dumping trace for {args.graph_name}, "
          f"{len(sampler)} batches/epoch x {args.num_epochs} epochs -> {out_dir}")

    for epoch in range(args.num_epochs):
        batch_nodes, batch_owners = [], []
        for input_nodes, seeds, blocks in sampler:
            with dist_lock:
                rem = input_nodes[~lmask[input_nodes]]
                if rem.numel() > 0:
                    owners = pb.nid2partid(rem.cpu())
                else:
                    owners = th.empty(0, dtype=th.long)
            batch_nodes.append(rem.cpu().numpy().astype(np.int64))
            batch_owners.append(owners.numpy().astype(np.int32))
        tr = Trace.from_batches(batch_nodes, batch_owners,
                                num_partitions=nparts, local_rank=pid)
        path = os.path.join(out_dir, f"r{pid}_epoch_{epoch:03d}.npz")
        tr.save(path)
        print(f"Part {pid}: epoch {epoch} -> {path}  "
              f"(B={tr.num_batches}, mean remote/batch={tr.mean_receptive_remote():.0f})")

    print(f"Part {pid}: trace dump complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, t, d in [
        ("--graph_name", str, None), ("--ip_config", str, None),
        ("--part_config", str, None), ("--backend", str, "gloo"),
        ("--num_epochs", int, 5), ("--fan_out", str, "10,25"),
        ("--batch_size", int, 2000), ("--out", str, "traces"),
        ("--seed", int, 1), ("--local_rank", int, None),
    ]:
        p.add_argument(a, type=t, default=d)
    main(p.parse_args())
