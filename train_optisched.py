#!/usr/bin/env python3
"""OptiSched-GNN trainer: non-uniform DP schedules + no-regret online control.

Drop-in sibling of train_greendygnn.py.  It reuses the same sampler / cache /
double-buffered prefetcher / energy + profiler scaffolding, but replaces the
Double-DQN with:

  * an offline Theorem-1 DP schedule per epoch (non-uniform windows), and
  * a fixed-share (Herbster-Warmuth) controller that, each epoch, picks which
    precomputed regime schedule to run and updates on full-information losses
    (Theorem 5) -- no training, no sim-to-real gap, a regret guarantee.

Modes (choose by which artifact you pass; build them with build_library.py):
  --library  lib.json    ONLINE fixed-share over regime experts (the system).
  --schedule sched.json  CLAIRVOYANT fixed per-epoch DP schedule (offline ceiling).
  (neither)              falls back to static window_size (== RapidGNN behaviour),
                         so the trainer always runs.

Everything that touches the hot path is unchanged from GreenDyGNN; the only added
per-epoch work is selecting a window-length list (microseconds) and one O(N)
controller update.
"""

import argparse, json, os, threading, time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl, dgl.distributed

from prefetcher import BatchPrefetcher, SharedBuffer, BackgroundSampler
from cache import FeatureCache
from energy_monitor import AccurateEnergyMonitor, CPUEnergyMonitor
from model import DistSAGE
from helpers import set_gpu_frequency, print_summary
from sampler import DistSampler
from presample import presample_and_cache
from metrics import TrainingProfiler, compute_accuracy
from optisched.regime import SimpleFixedShare


def estimate_sigma_hat(cache, recent_fetch, baseline_fetch, num_partitions,
                       local_rank, clamp=(1.0, 21.0)):
    """Runtime congestion estimate (proposal A7 / GreenDyGNN Eq. 8 analogue).

    We do not have per-owner RPC latency, so -- exactly as GreenDyGNN infers
    congestion from miss share -- we scale the measured overall fetch inflation
    by each owner's share of the recent misses.  Returns a length-P multiplier
    vector (local entry = 1).
    """
    sigma = np.ones(num_partitions)
    counts = cache.get_owner_miss_counts()
    if not counts or baseline_fetch is None or baseline_fetch <= 0:
        return sigma
    total = sum(counts.values()) or 1
    inflation = max(0.0, recent_fetch / baseline_fetch - 1.0)
    nrem = max(1, num_partitions - 1)
    for pid, c in counts.items():
        if 0 <= pid < num_partitions and pid != local_rank:
            share = (c / total) * nrem        # 1.0 == its fair share
            sigma[pid] = float(np.clip(1.0 + inflation * share, *clamp))
    return sigma


def nearest_regime(sigma_hat, sigmas):
    d = [float(np.linalg.norm(np.asarray(sigma_hat) - np.asarray(s))) for s in sigmas]
    return int(np.argmin(d))


def load_artifact(args):
    """Return (mode, data). mode in {'library','clairvoyant','static'}."""
    if args.library:
        with open(args.library) as f:
            return "library", json.load(f)
    if args.schedule:
        with open(args.schedule) as f:
            return "clairvoyant", json.load(f)
    return "static", None


def main(args):
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
        sl = g.ndata["labels"][train_nid[:min(10000, train_nid.numel())]]
        v = th.logical_and(~th.isnan(sl), sl >= 0)
        lm = th.max(sl[v]).long()
        th.distributed.all_reduce(lm, op=th.distributed.ReduceOp.MAX)
        args.n_classes = int(lm.item()) + 1

    pid = g.rank()
    nparts = th.distributed.get_world_size()
    mode, art = load_artifact(args)
    label = {"library": "optisched", "clairvoyant": "optisched_dp",
             "static": "optisched_static"}[mode]
    print(f"Part {pid}: OptiSched mode={mode}")

    profiler = TrainingProfiler(label, pid, output_dir=args.out_dir)
    gpu_mon = AccurateEnergyMonitor(device_index=dev_idx, tick=0.05)
    cpu_mon = CPUEnergyMonitor(verbose=False)
    gpu_mon.start(); cpu_mon.start()

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
    sampler = DistSampler(g, train_nid, args.fan_out, args.batch_size)
    bg = BackgroundSampler(sampler, sbuf, g, lmask, start_batch_id=0,
                           dist_lock=dist_lock, num_epochs=args.num_epochs)
    bg.start()
    bpe = len(sampler)
    tot = bpe * args.num_epochs

    ew = args.window_size if args.window_size > 0 else bpe
    sim_cache, cache, ps_time, _ = presample_and_cache(
        args, g, sbuf, device, dist_lock, max_batches=max(1, min(2 * ew, bpe)))
    set_gpu_frequency("default", dev_idx)

    W = args.window_size
    print(f"Part {pid}: {label} {args.graph_name} W0={W} cache={args.cache_size} bpe={bpe}")

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

    # Floor-guided owner-aware allocation (Thm G). Default off => uniform cache.
    if getattr(args, "owner_kappa", ""):
        ks = [float(x) for x in args.owner_kappa.split(",") if x != ""]
        pf.owner_kappa = {m: ks[m] for m in range(len(ks))}
        label = label + "_alloc"
        profiler.method = label
        print(f"Part {pid}: floor-guided owner-aware allocation kappa={pf.owner_kappa}")

    # --- controller / schedule setup ---
    ctl = None
    sigmas = None
    loss_tensor = None
    schedules = None
    wpe = None
    if mode == "library":
        sigmas = [np.asarray(s) for s in art["sigmas"]]
        loss_tensor = np.asarray(art["loss_tensor"])   # [N][N][E]
        schedules = art["schedules"]                   # [N][E] -> [w...]
        N = len(sigmas)
        E = loss_tensor.shape[2]
        ctl = SimpleFixedShare(N, horizon=args.num_epochs,
                               eps_mdl=art.get("model", {}).get("eps_mdl", 0.028))
        print(f"Part {pid}: library with {N} experts, {E} epochs of schedules")
    elif mode == "clairvoyant":
        wpe = art["window_lengths_per_epoch"]
        E = len(wpe)
        print(f"Part {pid}: clairvoyant schedule, {E} epochs")

    baseline_fetch = None
    fetch_window = []

    for epoch in range(args.num_epochs):
        te = time.time()
        e_losses, e_accs = [], []

        # ---- per-epoch scheduling decision (replaces DQN inference) ----
        if mode == "library":
            e_idx = min(epoch, loss_tensor.shape[2] - 1)
            expert = ctl.select()
            sched = schedules[expert][e_idx]
            pf.set_schedule(sched)
        elif mode == "clairvoyant":
            sched = wpe[min(epoch, len(wpe) - 1)]
            pf.set_schedule(sched)
        # static mode: leave pf.schedule None -> fixed window_size

        pf.start_epoch(epoch)
        fetch_window.clear()
        with ddp.join():
            for step in range(bpe):
                t0 = time.perf_counter()
                inp, lab, blk = pf.get()
                ft = cache.get_last_fetch_time()
                if ft > 0:
                    fetch_window.append(ft)
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
                cur_w = pf._win_len(pf.current_batch_idx) if pf.schedule else W
                profiler.record_step(epoch, step, loss.item(), acc, st,
                                     max(0, ft), gj, cj,
                                     cache_hit_pct=hr, extra={"W": cur_w})
                e_losses.append(loss.item())
                e_accs.append(acc)
                if (step + 1) % args.log_every == 0:
                    print(f"Part {pid} Ep{epoch:02d} S{step+1:3d}: "
                          f"L={loss.item():.4f} A={acc:.3f} W={cur_w}")

        # ---- end-of-epoch controller update ----
        recent_fetch = float(np.median(fetch_window[-30:])) if fetch_window else 0.0
        if epoch == 2 and fetch_window:
            baseline_fetch = float(np.percentile(fetch_window, 15))
        if mode == "library" and epoch >= 3 and baseline_fetch:
            sigma_hat = estimate_sigma_hat(cache, recent_fetch, baseline_fetch,
                                           nparts, pid)
            q = nearest_regime(sigma_hat, sigmas)
            e_idx = min(epoch, loss_tensor.shape[2] - 1)
            ctl.update_with_losses(loss_tensor[:, q, e_idx])
            if (epoch % args.log_every == 0) or epoch == args.num_epochs - 1:
                print(f"Part {pid} Ep{epoch:02d} ctl: sigma_hat~regime[{q}] "
                      f"{ctl.stats()}")

        et = time.time() - te
        gj = gpu_mon.get_total_gpu_energy()
        cj = cpu_mon.get_total_cpu_energy()
        profiler.record_epoch(epoch, et, gj, cj,
                              avg_loss=np.mean(e_losses),
                              avg_accuracy=np.mean(e_accs))
        print(f"Part {pid} Ep{epoch:02d}: {et:.2f}s GPU={gj:.1f}J "
              f"loss={np.mean(e_losses):.4f} acc={np.mean(e_accs):.3f}")
        if pf.owner_kappa and pf.last_owner_budgets is not None:
            print(f"Part {pid} Ep{epoch:02d} alloc budgets n_m={pf.last_owner_budgets}")

    gj = gpu_mon.get_total_gpu_energy()
    cj = cpu_mon.get_total_cpu_energy()
    print_summary(cache, pid, args, ps_time, gj, cj)
    if ctl is not None:
        print(f"Part {pid}: controller final {ctl.stats()}")
    profiler.save()
    gpu_mon.stop(); cpu_mon.stop()


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
    ]:
        p.add_argument(a, type=t, default=d)
    p.add_argument("--sync_cache", action="store_true")
    p.add_argument("--library", default="", help="regime-library JSON (online mode)")
    p.add_argument("--schedule", default="", help="clairvoyant schedule JSON")
    p.add_argument("--owner_kappa", default="",
                   help="floor-guided owner-aware allocation (Thm G): comma list of "
                        "per-owner relative costs kappa_m, e.g. '1,1,1,16'. Default off "
                        "(uniform top-frequency cache).")
    main(p.parse_args())
