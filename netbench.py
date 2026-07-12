#!/usr/bin/env python3
"""Per-owner NETWORK feature-fetch benchmark (measured owner heterogeneity).

Measures the time to pull N feature rows (dim d) from each owner node over the
private network, so a real throttle (tc netem) on one owner shows up as a higher
per-row time -> higher measured kappa[m] (under A4, energy ~ time).  This is the
measured asymmetry the allocation lever needs (not synthetic kappa x5).

server (on each owner node):   python3 netbench.py --server --port 31000 --d 128
client (on gnn1):              python3 netbench.py --client --hosts h1,h2,h3 \
                                   --port 31000 --d 128 --out per_owner.json
"""
import argparse, json, socket, struct, time, numpy as np


def server(a):
    buf = np.ascontiguousarray(np.random.standard_normal((a.maxN, a.d)).astype(np.float32)).tobytes()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", a.port)); s.listen(8)
    print(f"netbench server on :{a.port} d={a.d} maxN={a.maxN}", flush=True)
    while True:
        c, _ = s.accept()
        try:
            while True:
                hdr = c.recv(8)
                if not hdr or len(hdr) < 8:
                    break
                N = struct.unpack("!q", hdr)[0]
                if N <= 0:
                    break
                nbytes = N * a.d * 4
                c.sendall(buf[:nbytes])
        except Exception:
            pass
        finally:
            c.close()


def fetch(host, port, N, d):
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(30); sk.connect((host, port))
    nbytes = N * d * 4
    sk.sendall(struct.pack("!q", N))
    got = 0
    while got < nbytes:
        chunk = sk.recv(min(1 << 20, nbytes - got))
        if not chunk:
            break
        got += len(chunk)
    sk.close()
    return got


def client(a):
    hosts = a.hosts.split(",")
    sizes = [1, 100, 1000, 10000, 50000]
    sizes = [n for n in sizes if n <= a.maxN]
    out = {}
    for h in hosts:
        times = []
        for N in sizes:
            fetch(h, a.port, N, a.d)  # warmup
            t0 = time.perf_counter()
            for _ in range(a.reps):
                fetch(h, a.port, N, a.d)
            per = (time.perf_counter() - t0) / a.reps
            times.append(per)
        Nv = np.array(sizes, float)
        X = np.column_stack([np.ones_like(Nv), Nv])
        b, *_ = np.linalg.lstsq(X, np.array(times), rcond=None)
        t_init = max(times[0] - b[1] * sizes[0], 1e-9)   # small-N robust intercept
        t_slope = max(b[1], 1e-12)
        out[h] = {"t_init": float(t_init), "t_slope": float(t_slope),
                  "times": dict(zip(map(str, sizes), [float(x) for x in times]))}
        print(f"  {h}: t_init={t_init*1e3:.4f} ms  t_slope={t_slope*1e9:.2f} ns/row  "
              f"({t_slope*a.d*4/1e-9:.0f} ... {1/(t_slope/(4*a.d)+1e-30)/1e6:.0f} MB/s eff)", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--server", action="store_true"); p.add_argument("--client", action="store_true")
    p.add_argument("--hosts", default=""); p.add_argument("--port", type=int, default=31000)
    p.add_argument("--d", type=int, default=128); p.add_argument("--maxN", type=int, default=50000)
    p.add_argument("--reps", type=int, default=10); p.add_argument("--out", default="per_owner.json")
    a = p.parse_args()
    server(a) if a.server else client(a)
