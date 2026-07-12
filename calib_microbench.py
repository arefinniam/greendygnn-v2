#!/usr/bin/env python3
"""First-pass calibration of the communication-energy FLOOR coefficients via a
feature-fetch microbenchmark (cluster-side; needs RAPL readable).

Measures the energy + time of pulling N feature rows (dim d_bench) from a CPU
feature store -- the dominant cost in DistDGL remote feature fetch -- and fits
   time(N)   = t_init   + t_slope·N
   energy(N) = eps_init + (kappa·d_bench)·N
giving the floor inputs:  eps_init (J/RPC), kappa = e_slope/d_bench (J/feature),
t_miss ≈ t_slope (s/row).  C_max = largest stable N.  This calibrates the FLOOR
(eps_init, kappa, d, C_max) -- the trustworthy energy lens; the time-model rebuild
table (a,b,c) still needs the prefetcher and is left at defaults (labelled).

Writes a base calib.json (d=d_bench); make per-dataset copies with the dataset's d.
"""
import argparse, glob, json, os, time, numpy as np


def rapl_paths():
    return sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj"))


def read_rapl(paths):
    tot = 0
    for p in paths:
        try:
            tot += int(open(p).read().strip())
        except Exception:
            pass
    return tot  # microjoules


def main(a):
    rng = np.random.default_rng(0)
    d = a.d_bench
    store = np.asarray(rng.standard_normal((a.store_rows, d), dtype=np.float32))
    paths = rapl_paths()
    if not paths:
        print("[warn] no readable RAPL domains; energy will be 0 (time still valid)")

    sizes = [1, 10, 100, 1000, 5000, 20000, 50000]
    sizes = [n for n in sizes if n <= a.store_rows]
    times, energies = [], []
    for n in sizes:
        idx = rng.integers(0, a.store_rows, size=n)
        # warmup
        _ = store[idx].copy()
        e0 = read_rapl(paths); t0 = time.perf_counter()
        for _ in range(a.reps):
            idx = rng.integers(0, a.store_rows, size=n)
            out = store[idx].copy()           # gather = remote feature pull
            out += 0.0
        t1 = time.perf_counter(); e1 = read_rapl(paths)
        per_t = (t1 - t0) / a.reps
        per_e = (e1 - e0) / 1e6 / a.reps if paths else 0.0   # Joules per fetch
        times.append(per_t); energies.append(per_e)
        print(f"  N={n:6d}  time={per_t*1e3:.4f} ms  energy={per_e*1e3:.4f} mJ")

    N = np.array(sizes, float)
    # LS fit y = b0 + b1*N
    def fit(y):
        X = np.column_stack([np.ones_like(N), N])
        b, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
        return float(b[0]), float(b[1])
    t_init, t_slope = fit(times)
    e_init, e_slope = fit(energies)
    t_miss = max(t_slope, 1e-12)
    c_max = float(sizes[-1])      # largest stable fetch observed
    # RAPL energy_uj is root-only on many hardened kernels -> energy reads ~0.
    # Fall back to TIMING-based floor coefficients: under A4 (E ~ P_bar*time), the
    # energy coefficients are P_bar * time coefficients, so the floor RATIO
    # rho = E_win/L_0 is P_bar-INVARIANT and exactly calibrated from timing.
    energy_ok = e_slope > 0 and max(energies) > 0
    if energy_ok:
        eps_init = max(e_init, 1e-12); kappa = max(e_slope / d, 1e-18); basis = "RAPL-energy"
    else:
        # LS intercept is swamped by large-N points; the per-RPC init cost is best
        # read from the smallest fetch: time(N=1) - t_slope*1.
        t_init_small = max(times[0] - t_slope * sizes[0], t_slope)  # >=1 row-equiv
        eps_init = t_init_small; kappa = max(t_slope / d, 1e-18)
        basis = "timing (A4: rho is P_bar-invariant)"
    P = a.parts
    model = {
        "num_partitions": P, "d": d, "w_max": 64,
        "eps_init": [eps_init] * P, "kappa": [kappa] * P, "c_max": [c_max] * P,
        "t_miss": [t_miss] * P,
        "a": 0.002, "b": 0.0012, "c": 0.6, "alpha": 0.35,
        "p_bar": 190.0, "t_base": 0.008, "eps_mdl": 0.028,
        "_calibration": f"first-pass feature-fetch microbench; basis={basis}; FLOOR "
                        f"coeffs (eps_init,kappa,d,C_max); rebuild a,b,c are DEFAULT",
    }
    with open(a.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nFLOOR calibration (d_bench={d}, basis={basis}):")
    print(f"  eps_init = {eps_init:.4g} J/RPC   kappa = {kappa:.4g} J/feature   "
          f"kappa*d = {kappa*d:.4g} J/row")
    print(f"  t_miss(per-row) = {t_miss:.4g} s   C_max = {c_max:.0f} rows")
    print(f"  payload crossover n* = eps_init/(kappa*d) = {eps_init/(kappa*d):.0f} rows")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d_bench", type=int, default=128)
    p.add_argument("--store_rows", type=int, default=2000000)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--out", default="calib.json")
    main(p.parse_args())
