"""GPU frequency control, seeding, energy-monitor helpers, and summary printing."""

import random
import subprocess
import sys


def set_all_seeds(seed):
    """Seed random, numpy, torch (+CUDA) and return a torch.Generator (I6).

    Every trainer calls this with --seed so batch order, model init, and
    dropout are reproducible per (method, seed).
    """
    import numpy as np
    import torch as th
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    # Seed DGL explicitly (trace-equivalence blocker 2026-07-13): without it
    # the neighbor sampler's RNG is only indirectly covered and trace
    # collection vs live training identity cannot be guaranteed. Lazy +
    # guarded: unit tests run dgl-free.
    try:
        import dgl
        try:
            dgl.seed(seed)
        except AttributeError:
            dgl.random.seed(seed)
    except Exception:
        # dgl absent OR its native libs unloadable (e.g. LD_LIBRARY_PATH not
        # set — raises OSError, not ImportError). Trainers always import dgl
        # first, so when sampling actually happens this seed call succeeds.
        pass
    g = th.Generator()
    g.manual_seed(seed)
    return g


def sanitize_labels(labels, n_classes=None):
    """NaN-safe label sanitation shared by ALL trainers (invalid -> -1).

    Mirrors BatchPrefetcher._sanitize_labels so baselines (default/BGL) use
    the same substrate as cached methods and CrossEntropyLoss(ignore_index=-1)
    never sees garbage class indices from NaN->long casts.
    """
    import torch as th
    if labels.is_floating_point():
        labels = th.nan_to_num(labels, nan=-1.0)
    labels = labels.long()
    if n_classes is not None:
        labels = th.clamp(labels, min=-1, max=n_classes - 1)
    else:
        labels = th.clamp(labels, min=-1, max=1000)
    return labels


def estimate_n_classes(g, train_nid):
    """Unified n_classes estimation (max valid label + 1, all-reduced MAX).

    Used by every trainer when --n_classes is 0 so all four methods build
    identical output layers (review fairness fix).
    """
    import torch as th
    sl = g.ndata["labels"][train_nid[:min(10000, train_nid.numel())]]
    v = th.logical_and(~th.isnan(sl), sl >= 0)
    lm = th.max(sl[v]).long()
    th.distributed.all_reduce(lm, op=th.distributed.ReduceOp.MAX)
    return int(lm.item()) + 1


def set_gpu_frequency(mode="default", device_index=None):
    """Set GPU clock via nvidia-smi. mode='min' locks to lowest graphics clock;
    mode='default' resets. Used to suppress GPU energy draw during presampling.
    """
    try:
        gpu_flag = f"-i {device_index}" if device_index is not None else ""
        if mode == "min":
            result = subprocess.run(
                f"nvidia-smi {gpu_flag} --query-supported-clocks=gr "
                "--format=csv,noheader,nounits".split(),
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                clocks = [int(c.strip()) for c in result.stdout.strip().split('\n')
                          if c.strip().isdigit()]
                if clocks:
                    min_clock = min(clocks)
                    subprocess.run(
                        f"nvidia-smi {gpu_flag} -lgc {min_clock},{min_clock}".split(),
                        capture_output=True, timeout=5)
                    print(f"[GPU Freq] Set GPU to minimum frequency: {min_clock} MHz")
                    return min_clock
        else:
            subprocess.run(f"nvidia-smi {gpu_flag} -rgc".split(),
                           capture_output=True, timeout=5)
            print("[GPU Freq] Reset GPU to default frequency")
    except Exception as e:
        print(f"[GPU Freq] Warning: Could not set GPU frequency: {e}")
    return None


class MultiGPUEnergyMonitor:
    """Sum NVML energy across ALL visible GPUs on this node (spec P8).

    The second (idle) P100 per node draws ~25-30 W that 'total system
    energy' must include. Presents the same interface as
    AccurateEnergyMonitor (start/stop/get_total_gpu_energy).
    NOTE: run at most one instance per node (1 trainer/node layout) or the
    idle GPUs get double-counted; run_matrix.sh uses 1 trainer per node.
    """

    def __init__(self, tick=0.05, scope="all", device_index=None):
        from energy_monitor import AccurateEnergyMonitor
        self.monitors = []
        if scope == "all":
            try:
                import pynvml
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                print(f"[MultiGPU Monitor] NVML unavailable ({e}); "
                      f"falling back to current device only", file=sys.stderr)
                count = None
            if count:
                for i in range(count):
                    self.monitors.append(
                        AccurateEnergyMonitor(device_index=i, tick=tick))
        if not self.monitors:
            self.monitors.append(
                AccurateEnergyMonitor(device_index=device_index, tick=tick))

    def start(self):
        for m in self.monitors:
            m.start()

    def stop(self):
        for m in self.monitors:
            m.stop()

    def get_total_gpu_energy(self):
        return sum(m.get_total_gpu_energy() for m in self.monitors)

    def per_device(self):
        return [m.get_total_gpu_energy() for m in self.monitors]


def check_cpu_monitor(cpu_mon, part_id):
    """Fail-loud RAPL health check (spec P8).

    Returns True if the CPU monitor found readable RAPL domains. When it
    did not, the dominant energy component of every result would silently
    read 0.0 — print an unmissable warning on stderr; callers record
    cpu_energy_valid=False in the profiler.
    """
    ok = bool(getattr(cpu_mon, "cpu_ok", False)) and \
        len(getattr(cpu_mon, "rapl_domains", [])) > 0
    if not ok:
        msg = (f"!!! Part {part_id}: NO READABLE RAPL DOMAINS — CPU energy "
               f"will read 0.0 J. Run the RAPL chmod barrier "
               f"(sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj) "
               f"before launching. Result is INVALID for energy claims. !!!")
        print(msg, file=sys.stderr)
        print(msg)
    return ok


def print_summary(cache, part_id, args, presample_time, total_gpu_energy,
                  total_cpu_energy, method=None, seed=None, extra=None):
    """Per-partition summary. Prints BOTH the legacy lines (existing
    aggregation tooling greps 'Total GPU energy consumed:') AND one
    machine-readable SUMMARY line shared by all four trainers (spec P6).
    """
    hit_rate = 0.0
    remote_hits = remote_misses = 0
    fetch_ops = fetch_rows = fetch_bytes = 0
    if cache is not None:
        stats = cache.get_stats()
        hit_rate = stats['remote_cache_hit_rate'] * 100
        remote_hits = stats['remote_cache_hits']
        remote_misses = stats['remote_misses']
        if hasattr(cache, "get_remote_fetch_counters"):
            fetch_ops, fetch_rows, fetch_bytes = cache.get_remote_fetch_counters()
        print(f"Part {part_id}: Remote cache hit rate {hit_rate:.1f}% "
              f"(hits: {remote_hits}, misses: {remote_misses}), "
              f"WS={getattr(args, 'window_size', 0)}, Cache={getattr(args, 'cache_size', 0)}")
    print(f"Part {part_id}: Total presample time: {presample_time:.2f}s")
    print(f"Part {part_id}: Total GPU energy consumed: {total_gpu_energy:.2f}J")
    print(f"Part {part_id}: Total CPU energy consumed: {total_cpu_energy:.2f}J")
    print(f"Part {part_id}: Total energy consumed: {total_gpu_energy + total_cpu_energy:.2f}J")

    method = method or getattr(args, "method_label", "unknown")
    seed = seed if seed is not None else getattr(args, "seed", 0)
    fields = {
        "method": method,
        "part": part_id,
        "seed": seed,
        "gpu_j": f"{total_gpu_energy:.2f}",
        "cpu_j": f"{total_cpu_energy:.2f}",
        "total_j": f"{total_gpu_energy + total_cpu_energy:.2f}",
        "presample_s": f"{presample_time:.2f}",
        "hit_pct": f"{hit_rate:.2f}",
        "remote_hits": remote_hits,
        "remote_misses": remote_misses,
        "remote_fetch_ops": fetch_ops,
        "remote_rows": fetch_rows,
        "remote_bytes": fetch_bytes,
        "ws": getattr(args, "window_size", 0),
        "cache": getattr(args, "cache_size", 0),
        "batch_size": getattr(args, "batch_size", 0),
    }
    if extra:
        fields.update(extra)
    print("SUMMARY " + " ".join(f"{k}={v}" for k, v in fields.items()))
