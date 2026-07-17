#!/usr/bin/env python3
"""Collect per-batch remote-access traces (RESEARCH_PLAN_v2 item 3, Layer 1).

Runs the PRODUCTION sampling path (same DistSampler, same set_all_seeds — which
now seeds DGL explicitly — same node_split) with no training, and dumps every
batch's remote node IDs + owners in the Layer-1 engine's native format:
one optisched.trace.Trace CSR file per epoch per rank.

Trace-EXACTNESS is checkable, not asserted: the collector folds every batch
through trace_digest (global batch index + remote int64 ids, BEFORE storage
dtype conversion), and any training run launched with --trace_digest and the
same (dataset, partitioning, B, fanout, seed) writes the same rolling digest
from its live BackgroundSampler. Equal digests == byte-identical streams.
Verification protocol (run before trusting any trace):
  1. collect the same trace twice -> digests must match (determinism);
  2. run one real training with --trace_digest -> must match the collector.

Launch exactly like a trainer (one rank per partition):
  python3 collect_trace.py --graph_name reddit --ip_config ... --part_config \
      ... --batch_size 2000 --fan_out 10,25 --num_epochs 30 --seed 0 --out_dir ...

Output per rank:
  {out_dir}/trace_part{pid}_ep{e:03d}.npz   optisched.trace.Trace CSR
  {out_dir}/trace_part{pid}_meta.json       config + digest + fingerprints
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch as th
import dgl
import dgl.distributed

from helpers import set_all_seeds
from sampler import DistSampler
from trace_digest import new_digest, update_digest
from optisched.trace import Trace


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
    P = pb.num_partitions()

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
    # partition fingerprint over the DATA, not just the config file: local
    # inner-node id set identifies this partitioning assignment.
    local_part_md5 = hashlib.md5(
        lid.sort().values.numpy().astype("<i8").tobytes()).hexdigest()

    sampler = DistSampler(g, train_nid, args.fan_out, args.batch_size,
                          generator=gen)
    bpe = len(sampler)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Part {pid}: tracing {args.graph_name} bpe={bpe} "
          f"epochs={args.num_epochs} seed={args.seed}")

    digest = new_digest()
    global_batch = 0
    t0 = time.time()
    epoch_files = []
    for epoch in range(args.num_epochs):
        batch_nodes, batch_owners = [], []
        for input_nodes, _seeds, _blocks in sampler:
            remote = input_nodes[~lmask[input_nodes]]
            update_digest(digest, global_batch, remote)
            global_batch += 1
            owners = pb.nid2partid(remote)
            batch_nodes.append(remote.numpy().astype(np.int64))
            batch_owners.append(owners.numpy().astype(np.int32))
        total = int(sum(len(x) for x in batch_nodes))
        tr = Trace(
            nodes=np.concatenate(batch_nodes) if total else
            np.empty(0, np.int64),
            owners=np.concatenate(batch_owners) if total else
            np.empty(0, np.int32),
            offsets=np.cumsum([0] + [len(x) for x in batch_nodes],
                              dtype=np.int64),
            num_partitions=P, local_rank=int(pid))
        path = os.path.join(args.out_dir,
                            f"trace_part{pid}_ep{epoch:03d}.npz")
        tr.save(path)
        epoch_files.append(os.path.basename(path))
        print(f"Part {pid}: epoch {epoch} -> {path} "
              f"({len(batch_nodes)} batches, {total} ids, "
              f"{time.time() - t0:.0f}s)")

    with open(args.part_config, "rb") as f:
        cfg_md5 = hashlib.md5(f.read()).hexdigest()
    meta = dict(
        dataset=args.graph_name, part_config=args.part_config,
        part_config_md5=cfg_md5, local_partition_md5=local_part_md5,
        num_nodes=int(g.num_nodes()), P=int(P), pid=int(pid),
        feat_dim=int(g.ndata["features"].shape[1]),
        feat_dtype=str(g.ndata["features"].dtype),
        batch_size=args.batch_size, fan_out=args.fan_out,
        num_epochs=args.num_epochs, bpe=bpe, seed=args.seed,
        n_train_local=int(train_nid.numel()),
        n_batches=global_batch, digest=digest.hexdigest(),
        epoch_files=epoch_files, collected_t=time.time(),
    )
    mpath = os.path.join(args.out_dir, f"trace_part{pid}_meta.json")
    json.dump(meta, open(mpath, "w"), indent=2)
    print(f"Part {pid}: digest={meta['digest']} -> {mpath}")
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
