#!/usr/bin/env python3
"""Default DGL baseline: on-demand feature fetching, no caching or prefetching."""

import argparse, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl, dgl.distributed

from model import DistSAGE
from energy_monitor import CPUEnergyMonitor
from helpers import (print_summary, set_all_seeds, sanitize_labels,
                     estimate_n_classes, MultiGPUEnergyMonitor,
                     check_cpu_monitor)
from metrics import TrainingProfiler, compute_accuracy


def main(args):
    gen = set_all_seeds(args.seed)
    dgl.distributed.initialize(args.ip_config)
    th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    pb = g.get_partition_book()
    train_nid = dgl.distributed.node_split(g.ndata["train_mask"], pb, force_even=True)

    device = th.device(f"cuda:{g.rank() % args.num_gpus}") if args.num_gpus else th.device("cpu")
    if args.num_gpus:
        th.cuda.set_device(device)
    dev_idx = device.index if args.num_gpus else None

    n_classes = args.n_classes
    if n_classes == 0:
        n_classes = estimate_n_classes(g, train_nid)

    pid = g.rank()
    profiler = TrainingProfiler("default_dgl", pid, output_dir=args.out_dir)
    profiler.set_meta(seed=args.seed, label="default_dgl", graph=args.graph_name,
                      batch_size=args.batch_size, n_classes=n_classes)

    gpu_mon = MultiGPUEnergyMonitor(tick=0.05, scope=args.gpu_energy_scope,
                                    device_index=dev_idx)
    cpu_mon = CPUEnergyMonitor(verbose=False)
    gpu_mon.start(); cpu_mon.start()
    cpu_valid = check_cpu_monitor(cpu_mon, pid)
    profiler.set_meta(cpu_energy_valid=bool(cpu_valid))

    fr = None
    if args.flight_recorder:
        import os
        from flight_recorder import FlightRecorder
        os.makedirs(args.out_dir, exist_ok=True)
        fr = FlightRecorder(os.path.join(args.out_dir,
                                         f"flight_part{pid}.jsonl"),
                            gpu_index=dev_idx, rank=pid)
        fr.start()

    sampler = dgl.dataloading.NeighborSampler(
        [int(f) for f in args.fan_out.split(",")])
    try:
        dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, sampler, batch_size=args.batch_size,
            shuffle=True, drop_last=False, generator=gen)
    except TypeError:
        dataloader = dgl.dataloading.DistNodeDataLoader(
            g, train_nid, sampler, batch_size=args.batch_size,
            shuffle=True, drop_last=False)

    in_feats = g.ndata["features"].shape[1]
    model = DistSAGE(in_feats, args.num_hidden, n_classes,
                     args.num_layers, F.relu, args.dropout).to(device)
    ddp = th.nn.parallel.DistributedDataParallel(
        model, device_ids=[device] if args.num_gpus else None)
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    optimizer = optim.Adam(ddp.parameters(), lr=args.lr)

    print(f"Part {pid}: Default DGL {args.graph_name} B={args.batch_size} "
          f"seed={args.seed}")

    for epoch in range(args.num_epochs):
        tic = time.time()
        e_losses, e_accs = [], []

        with ddp.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                t0 = time.perf_counter()
                ft0 = time.perf_counter()
                batch_inputs = g.ndata["features"][input_nodes]
                batch_labels = sanitize_labels(g.ndata["labels"][seeds], n_classes)
                fetch_time = time.perf_counter() - ft0

                blocks = [b.to(device) for b in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                pred = ddp(blocks, batch_inputs)
                loss = loss_fcn(pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                st = time.perf_counter() - t0

                acc = compute_accuracy(pred.detach(), batch_labels)
                gj = gpu_mon.get_total_gpu_energy()
                cj = cpu_mon.get_total_cpu_energy()
                profiler.record_step(epoch, step, loss.item(), acc, st,
                                     fetch_time, gj, cj)
                e_losses.append(loss.item())
                e_accs.append(acc)

                if (step + 1) % args.log_every == 0:
                    print(f"Part {pid} Ep{epoch:02d} S{step+1:3d}: "
                          f"L={loss.item():.4f} A={acc:.3f}")

        et = time.time() - tic
        gj = gpu_mon.get_total_gpu_energy()
        cj = cpu_mon.get_total_cpu_energy()
        profiler.record_epoch(epoch, et, gj, cj,
                              avg_loss=np.mean(e_losses),
                              avg_accuracy=np.mean(e_accs))
        print(f"Part {pid} Ep{epoch:02d}: {et:.2f}s GPU={gj:.1f}J "
              f"loss={np.mean(e_losses):.4f} acc={np.mean(e_accs):.3f}")

    gj = gpu_mon.get_total_gpu_energy()
    cj = cpu_mon.get_total_cpu_energy()
    print_summary(None, pid, args, 0.0, gj, cj, method="default_dgl",
                  seed=args.seed, extra={"cpu_energy_valid": cpu_valid})
    profiler.save()
    gpu_mon.stop(); cpu_mon.stop()
    if fr:
        fr.stop()


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
        ("--local_rank", int, None), ("--out_dir", str, "logs"),
        ("--seed", int, 0), ("--gpu_energy_scope", str, "all"),
    ]:
        p.add_argument(a, type=t, default=d)
    p.add_argument("--flight_recorder", action="store_true",
                   help="1Hz node time series (NIC/CPU/RAPL/GPU) to out_dir")
    main(p.parse_args())
