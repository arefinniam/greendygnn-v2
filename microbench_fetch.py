#!/usr/bin/env python3
"""Production-path fetch microbenchmark (RESEARCH_PLAN_v2 item 3, Layer 2).

Measures per-owner feature-fetch RTT over a rows ladder through the REAL
DistTensor path (g.ndata["features"][nodes]) — the same code the cache's miss
path and builder use — so the fitted T_m(R) = a_m + b_m*R surfaces include
serialization, RPC framing, and DGL overheads, not just wire time.

Congestion conditions are applied EXTERNALLY (congestion2.py apply/run) by the
campaign driver; this script only measures and records. Run it like a trainer
(one rank per partition):

  python3 microbench_fetch.py --graph_name reddit --ip_config ... \
      --part_config ... --out_dir bench_out --tag clean

Modes:
  default        single-flow ladder: rows in --ladder, --reps pulls each,
                 per remote owner, randomized (owner, rows) order.
  --concurrent   additionally repeats the ladder while a background thread
                 does continuous bulk pulls (--bulk_rows) from the SAME rank —
                 the fetch x rebuild coupling cell of the concurrency matrix.

Output per rank: {out_dir}/fetchbench_{tag}_part{pid}.json
  {"meta": {...}, "samples": [{"owner", "rows", "rtt_s", "concurrent"}, ...]}
"""

import argparse
import json
import os
import random
import threading
import time

import torch as th
import dgl
import dgl.distributed

from helpers import set_all_seeds


def owner_nodes(pb, pid, n):
    nids = pb.partid2nids(pid)
    if not isinstance(nids, th.Tensor):
        nids = th.as_tensor(nids)
    idx = th.randint(0, nids.numel(), (min(n, nids.numel()),))
    return nids[idx].long()


def run_ladder(g, pb, local, ladder, reps, rng, concurrent=False, lock=None):
    remote_owners = [p for p in range(pb.num_partitions()) if p != local]
    plan = [(o, r) for o in remote_owners for r in ladder for _ in range(reps)]
    rng.shuffle(plan)
    samples = []
    for owner, rows in plan:
        nodes = owner_nodes(pb, owner, rows)
        t0 = time.perf_counter()
        if lock is not None:
            with lock:
                t1 = time.perf_counter()
                _ = g.ndata["features"][nodes]
                rtt = time.perf_counter() - t1
            lock_s = t1 - t0
        else:
            _ = g.ndata["features"][nodes]
            rtt = time.perf_counter() - t0
            lock_s = 0.0
        samples.append({"owner": int(owner), "rows": int(nodes.numel()),
                        "rtt_s": rtt, "lock_s": lock_s, "t": time.time(),
                        "concurrent": bool(concurrent)})
    return samples


def main(args):
    set_all_seeds(args.seed)
    rng = random.Random(args.seed)
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    pb = g.get_partition_book()
    pid = g.rank()
    local = getattr(pb, "partid", pid)
    ladder = [int(x) for x in args.ladder.split(",")]

    # warm the RPC path so first-connection setup doesn't pollute sample 1
    for p in range(pb.num_partitions()):
        if p != local:
            _ = g.ndata["features"][owner_nodes(pb, p, 8)]

    samples = run_ladder(g, pb, local, ladder, args.reps, rng)

    if args.concurrent:
        # The DGL distributed RPC client is NOT thread-safe: unguarded pulls
        # from two threads SIGABRT (proven 2026-07-16, all conc cells).
        # Production never does that — every DistTensor access is arbitrated
        # by dist_lock — so the coupling cell mirrors production: bulk thread
        # and ladder serialize through the same lock, and the ladder records
        # lock-wait (queue) separately from wire time, like the cache's
        # fetch-event split.
        stop = threading.Event()
        dist_lock = threading.Lock()

        def bulk():
            owners = [p for p in range(pb.num_partitions()) if p != local]
            i = 0
            while not stop.is_set():
                try:
                    nodes = owner_nodes(pb, owners[i % len(owners)],
                                        args.bulk_rows)
                    with dist_lock:
                        _ = g.ndata["features"][nodes]
                except Exception as e:
                    print(f"bulk thread error, stopping: {e}")
                    return
                i += 1

        t = threading.Thread(target=bulk, daemon=True)
        t.start()
        try:
            samples += run_ladder(g, pb, local, ladder, args.reps, rng,
                                  concurrent=True, lock=dist_lock)
        finally:
            stop.set()
            t.join(timeout=30)

    meta = dict(dataset=args.graph_name, pid=int(pid), local_owner=int(local),
                P=int(pb.num_partitions()), ladder=ladder, reps=args.reps,
                tag=args.tag, concurrent=bool(args.concurrent),
                bulk_rows=args.bulk_rows,
                feat_dim=int(g.ndata["features"].shape[1]),
                seed=args.seed, t=time.time())
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"fetchbench_{args.tag}_part{pid}.json")
    json.dump({"meta": meta, "samples": samples}, open(out, "w"))
    print(f"Part {pid}: {len(samples)} samples -> {out}")
    th.distributed.barrier()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, t, d in [
        ("--graph_name", str, None), ("--ip_config", str, None),
        ("--part_config", str, None), ("--backend", str, "gloo"),
        ("--ladder", str, "1,10,100,1000,10000,100000"),
        ("--reps", int, 15), ("--seed", int, 0), ("--tag", str, "clean"),
        ("--bulk_rows", int, 50000), ("--out_dir", str, "bench_out"),
        ("--local_rank", int, None), ("--num_gpus", int, 0),
        ("--n_classes", int, 0),
    ]:
        p.add_argument(a, type=t, default=d)
    p.add_argument("--concurrent", action="store_true")
    main(p.parse_args())
