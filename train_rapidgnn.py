#!/usr/bin/env python3
"""RapidGNN baseline: epoch-level presampling + static hot-feature cache.

--window_size 0  => TRUE RapidGNN (one cache build per epoch, W = batches
                    per epoch) — label `rapidgnn_epoch` (spec P9).
--window_size N  => static windowed variant (the paper's own "w/o RL"
                    mechanism at fixed W) — label `static_wN`.
"""

import argparse, os, threading, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl, dgl.distributed

from prefetcher import BatchPrefetcher, SharedBuffer, BackgroundSampler
from cache import FeatureCache
from energy_monitor import CPUEnergyMonitor
from model import DistSAGE
from helpers import (set_gpu_frequency, print_summary, set_all_seeds,
                     estimate_n_classes, MultiGPUEnergyMonitor,
                     check_cpu_monitor)
from sampler import DistSampler
from presample import presample_and_cache
from metrics import TrainingProfiler, compute_accuracy


def main(args):
    gen = set_all_seeds(args.seed)
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    train_nid = dgl.distributed.node_split(
        g.ndata["train_mask"], g.get_partition_book(), force_even=True)

    device = th.device(f"cuda:{g.rank() % args.num_gpus}") if args.num_gpus else th.device("cpu")
    if args.num_gpus:
        th.cuda.set_device(device)
    dev_idx = device.index if args.num_gpus else None

    if args.n_classes == 0:
        args.n_classes = estimate_n_classes(g, train_nid)

    pid = g.rank()
    label = "rapidgnn_epoch" if args.window_size == 0 else f"static_w{args.window_size}"
    profiler = TrainingProfiler(label, pid, output_dir=args.out_dir)
    profiler.set_meta(seed=args.seed, label=label, graph=args.graph_name,
                      batch_size=args.batch_size, window_size=args.window_size,
                      cache_size=args.cache_size)

    gpu_mon = MultiGPUEnergyMonitor(tick=0.05, scope=args.gpu_energy_scope,
                                    device_index=dev_idx)
    cpu_mon = CPUEnergyMonitor(verbose=False)
    gpu_mon.start(); cpu_mon.start()
    cpu_valid = check_cpu_monitor(cpu_mon, pid)
    profiler.set_meta(cpu_energy_valid=bool(cpu_valid))

    os.makedirs(args.out_dir, exist_ok=True)
    set_gpu_frequency("min", dev_idx)
    dist_lock = threading.Lock()

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

    sbuf = SharedBuffer(capacity=500)
    sampler = DistSampler(g, train_nid, args.fan_out, args.batch_size,
                          generator=gen)
    bg = BackgroundSampler(sampler, sbuf, g, lmask, start_batch_id=0,
                           dist_lock=dist_lock, num_epochs=args.num_epochs)
    bg.start()
    bpe = len(sampler)
    tot = bpe * args.num_epochs

    # W=0 => epoch-level cache: one rebuild per epoch (TRUE RapidGNN).
    W = args.window_size if args.window_size > 0 else bpe
    ew = W
    sim_cache, cache, ps_time, _ = presample_and_cache(
        args, g, sbuf, device, dist_lock, max_batches=max(1, min(2 * ew, bpe)))
    set_gpu_frequency("default", dev_idx)

    print(f"Part {pid}: {label} {args.graph_name} W={W} cache={args.cache_size} "
          f"bpe={bpe} seed={args.seed}")

    model = DistSAGE(g.ndata["features"].shape[1], args.num_hidden,
                     args.n_classes, args.num_layers, F.relu, args.dropout).to(device)
    ddp = th.nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if args.num_gpus else None)
    opt = optim.Adam(ddp.parameters(), lr=args.lr)
    lfn = nn.CrossEntropyLoss(ignore_index=-1).to(device)

    pf = BatchPrefetcher(g, device, cache, W, sbuf, tot, bpe,
                         max_batches=args.prefetch_buffer_size,
                         synchronous_cache=getattr(args, 'sync_cache', False),
                         n_classes=args.n_classes, window_hot_nodes={},
                         dist_lock=dist_lock, initial_data=sim_cache)

    for epoch in range(args.num_epochs):
        te = time.time()
        e_losses, e_accs = [], []
        pf.start_epoch(epoch)

        with ddp.join():
            for step in range(bpe):
                t0 = time.perf_counter()
                item = pf.get()
                if item is None:
                    raise RuntimeError(
                        f"Part {pid}: prefetcher returned no batch at "
                        f"epoch {epoch} step {step}")
                inp, lab, blk = item
                ft = cache.get_last_fetch_time()
                out = ddp(blk, inp)
                loss = lfn(out, lab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                st = time.perf_counter() - t0

                acc = compute_accuracy(out.detach(), lab)
                hr = cache.get_step_remote_hit_rate()
                gj = gpu_mon.get_total_gpu_energy()
                cj = cpu_mon.get_total_cpu_energy()
                f_ops, f_rows, f_bytes = cache.get_remote_fetch_counters()
                profiler.record_step(epoch, step, loss.item(), acc, st,
                                     max(0, ft), gj, cj, cache_hit_pct=hr,
                                     extra={"remote_fetch_ops": f_ops,
                                            "remote_rows": f_rows,
                                            "remote_bytes": f_bytes})
                e_losses.append(loss.item())
                e_accs.append(acc)

                if (step + 1) % args.log_every == 0:
                    print(f"Part {pid} Ep{epoch:02d} S{step+1:3d}: "
                          f"L={loss.item():.4f} A={acc:.3f} W={W}")

        et = time.time() - te
        gj = gpu_mon.get_total_gpu_energy()
        cj = cpu_mon.get_total_cpu_energy()
        profiler.record_rebuilds(pf.drain_rebuild_log())
        profiler.record_epoch(epoch, et, gj, cj,
                              avg_loss=np.mean(e_losses),
                              avg_accuracy=np.mean(e_accs),
                              extra={"owner_latency":
                                     cache.get_owner_latency_stats(last_n=1024),
                                     "health": pf.get_health_counters()})
        print(f"Part {pid} Ep{epoch:02d}: {et:.2f}s GPU={gj:.1f}J "
              f"loss={np.mean(e_losses):.4f} acc={np.mean(e_accs):.3f}")

    gj = gpu_mon.get_total_gpu_energy()
    cj = cpu_mon.get_total_cpu_energy()
    print_summary(cache, pid, args, ps_time, gj, cj, method=label,
                  seed=args.seed,
                  extra={"cpu_energy_valid": cpu_valid,
                         **pf.get_health_counters()})
    profiler.save()
    gpu_mon.stop(); cpu_mon.stop()
    pf.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for a, t, d in [
        ("--graph_name", str, None), ("--ip_config", str, None),
        ("--part_config", str, None), ("--n_classes", int, 0),
        ("--backend", str, "gloo"), ("--num_gpus", int, 1),
        ("--num_epochs", int, 10), ("--num_hidden", int, 16),
        ("--num_layers", int, 2), ("--fan_out", str, "10,25"),
        ("--batch_size", int, 1000), ("--log_every", int, 20),
        ("--lr", float, 0.003), ("--dropout", float, 0.5),
        ("--local_rank", int, None), ("--cache_size", int, 100000),
        ("--window_size", int, 16), ("--prefetch_buffer_size", int, 100),
        ("--presample_batches", int, 2000), ("--out_dir", str, "logs"),
        ("--seed", int, 0), ("--gpu_energy_scope", str, "all"),
    ]:
        p.add_argument(a, type=t, default=d)
    p.add_argument("--sync_cache", action="store_true")
    main(p.parse_args())
