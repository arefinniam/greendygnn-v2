#!/usr/bin/env python3
"""Node-local power-state map (RESEARCH_PLAN_v2 item 3, Layer 3).

Measures steady CPU (RAPL) and GPU (NVML) power in defined activity states so
the simulator can attribute energy as E = sum over phases of P(state)*t(state)
instead of the retired fetch-time*avg-power proxy:

  idle          nothing running
  cpu_spin      N busy CPU threads (RPC-serving / serialization proxy)
  gpu_compute   repeated large matmul (training-step proxy)
  h2d_copy      repeated 100 MB host->device copies (feature-upload proxy)

Each state runs for --hold seconds while sampling at 10 Hz via FlightRecorder,
then reports mean watts per source. Run standalone on each node (no cluster
coordination): python3 power_map.py --out power_map.json
"""

import argparse
import json
import os
import tempfile
import threading
import time


def _summarize(jsonl_path):
    """Mean CPU W (RAPL counter slope) and GPU W (NVML power) of one state."""
    lines = [json.loads(l) for l in open(jsonl_path)]
    samples = [l for l in lines if not l.get("meta")]
    out = {"n": len(samples), "cpu_w": None, "gpu_w": None}
    if len(samples) >= 2:
        rapl0, rapl1 = samples[0].get("rapl"), samples[-1].get("rapl")
        dt = samples[-1]["t"] - samples[0]["t"]
        if rapl0 and rapl1 and dt > 0:
            duj = sum(max(0, b - a) for a, b in zip(rapl0, rapl1)
                      if a is not None and b is not None)
            out["cpu_w"] = duj / 1e6 / dt
        pw = [g["p_mw"] for s in samples for g in (s.get("gpu") or []) if g]
        if pw:
            out["gpu_w"] = sum(pw) / len(pw) / 1000.0
    return out


def measure_state(name, load_fn, hold_s, iface="eno1"):
    """Run load_fn (returns a stop callable) for hold_s while recording."""
    from flight_recorder import FlightRecorder
    path = os.path.join(tempfile.gettempdir(), f"powermap_{name}.jsonl")
    stop_load = load_fn() if load_fn else None
    time.sleep(1.0)                       # let the state settle
    fr = FlightRecorder(path, interval_s=0.1, iface=iface)
    fr.start()
    time.sleep(hold_s)
    fr.stop()
    if stop_load:
        stop_load()
    res = _summarize(path)
    res["state"] = name
    return res


def _spin_proc(stop_event):
    x = 1.0
    while not stop_event.is_set():
        x = x * 1.0000001 + 1e-9


def cpu_spin_load(n_procs):
    # Processes, not threads: Python spin threads hold the GIL ~100% and starve
    # the FlightRecorder sampler thread (observed 1-4 samples per 20 s hold).
    def start():
        import multiprocessing as mp
        stop = mp.Event()
        ps = [mp.Process(target=_spin_proc, args=(stop,), daemon=True)
              for _ in range(n_procs)]
        [p.start() for p in ps]

        def stop_all():
            stop.set()
            for p in ps:
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()
        return stop_all
    return start


def gpu_load(kind, device_index=0):
    def start():
        import torch as th
        if not th.cuda.is_available():
            return None
        stop = threading.Event()
        dev = th.device(f"cuda:{device_index}")

        def work():
            if kind == "matmul":
                a = th.randn(4096, 4096, device=dev)
                while not stop.is_set():
                    a = a @ a * 1e-4
                    th.cuda.synchronize(dev)
            else:  # h2d
                host = th.randn(25_000_000)          # ~100 MB fp32
                while not stop.is_set():
                    _ = host.to(dev)
                    th.cuda.synchronize(dev)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        return lambda: (stop.set(), t.join(timeout=10))
    return start


def main(args):
    states = [
        ("idle", None),
        ("cpu_spin", cpu_spin_load(args.spin_threads)),
        ("gpu_compute", gpu_load("matmul", args.gpu_index)),
        ("h2d_copy", gpu_load("h2d", args.gpu_index)),
    ]
    results = []
    for name, load in states:
        print(f"[power_map] measuring {name} for {args.hold}s ...")
        try:
            results.append(measure_state(name, load, args.hold, args.iface))
        except Exception as e:
            results.append({"state": name, "error": f"{type(e).__name__}: {e}"})
        print(f"[power_map]   -> {results[-1]}")
    out = {"host": os.uname().nodename, "t": time.time(),
           "hold_s": args.hold, "spin_threads": args.spin_threads,
           "states": results}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[power_map] wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="power_map.json")
    p.add_argument("--hold", type=float, default=20.0)
    p.add_argument("--spin_threads", type=int, default=8)
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--iface", default="eno1")
    main(p.parse_args())
