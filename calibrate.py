#!/usr/bin/env python3
"""Calibrate the OptiSched CostModel (proposal A3, GreenDyGNN Alg. 1 analogue).

Produces a CostModel JSON (consumed by run_gate.py / build_library.py) from a
one-time profiling run.  Two inputs, either/both:

  --measurements meas.json   {
       "rebuild": {"windows": [1,2,4,8,16,32,64], "times_s": [...]},   # T_rebuild(w)
       "t_miss_s": {"1": 4.1e-4, "2": 3.9e-4, "3": 4.3e-4},           # per owner
       "alpha": 0.35, "p_bar": 190.0, "t_base": 8e-3, "eps_mdl": 0.028
     }
  --profile prof.json        a TrainingProfiler JSON; used to estimate p_bar
                             (mean active power = final gpu_energy / wall_time).

Anything not supplied falls back to the documented defaults.  The rebuild fit
(a + b*w**c) uses the no-SciPy grid+LS solver in calibration.CostModel.fit_rebuild.
"""

import argparse
import json

import numpy as np

from optisched.calibration import CostModel, default_model


def main(args):
    P = args.num_partitions
    m = default_model(num_partitions=P, w_max=args.w_max)

    if args.measurements:
        meas = json.load(open(args.measurements))
        if "rebuild" in meas:
            r = meas["rebuild"]
            a, b, c = m.fit_rebuild(r["windows"], r["times_s"])
            print(f"rebuild fit: a={a:.4g} b={b:.4g} c={c:.3g}")
        if "t_miss_s" in meas:
            per = {int(k): float(v) for k, v in meas["t_miss_s"].items()}
            m.fit_miss_latency(per)
            print(f"t_miss: {m.t_miss}")
        for k in ("alpha", "p_bar", "t_base", "eps_mdl"):
            if k in meas:
                setattr(m, k, float(meas[k]))

    if args.profile:
        prof = json.load(open(args.profile))
        wall = prof.get("total_wall_time_s", 0)
        gpu = prof["epochs"][-1]["gpu_energy_j"] if prof.get("epochs") else 0
        if wall and gpu:
            m.p_bar = float(gpu / wall)
            print(f"p_bar from profile: {m.p_bar:.1f} W "
                  f"(gpu {gpu:.0f} J / {wall:.0f} s)")

    m.save(args.out)
    print(f"CostModel -> {args.out}")
    print(json.dumps(m.to_dict(), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--measurements", default="", help="profiling measurements JSON")
    p.add_argument("--profile", default="", help="TrainingProfiler JSON (for p_bar)")
    p.add_argument("--num_partitions", type=int, default=4)
    p.add_argument("--w_max", type=int, default=64)
    p.add_argument("--out", default="calib.json")
    main(p.parse_args())
