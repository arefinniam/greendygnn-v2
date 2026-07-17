# Experiment results

Aggregated output from cluster campaigns run on gnn1-4. Each subdirectory holds
the small summary artifacts (`matrix.tsv`, `results_table.json/.md`, `plan.txt`)
for one campaign — not the raw per-run outputs.

Raw run directories (checkpoints, per-epoch logs, and trace dumps) stay on the
cluster under `~/matrix_results/<campaign>/runs/` — several GB per campaign,
not checked in here.

- `matrix_20260708_064408/` — v1 audit matrix (216 runs, 6 methods x 2 datasets
  x 6 conditions x 3 seeds) + `SESSION_HANDOFF_2026-07-10.md`.
- `wsweep_20260711_170421/` — Step-0 window-size sweep (176 runs).
- `transitions_20260715/` — scripted W-transition campaign (24 runs, r1-r3 x
  2 datasets x 4 steady conditions).
- `instr_overhead_20260715/`, `instr_baseline_20260716/`,
  `instr_abl_flight_recorder_20260717/`, `instr_abl_trace_digest_20260717/`,
  `instr_abl_trace_dump_20260717/`, `instr_abl_tracedump_fixed_20260717/`,
  `instr_abl_tracedump_v2_20260717/` — instrumentation-overhead ablation cells.
- `microbench_20260715/` — production-path fetch RTT-vs-rows microbenchmarks.
- `power_map_20260715/` — per-node idle/cpu_spin/gpu_compute/h2d power maps.
- `traces_smoke/` — smoke-test evidence for the trace-nondeterminism finding
  (digests + logs only; raw `.npz` trace dumps excluded).
