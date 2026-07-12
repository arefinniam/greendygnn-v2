# OptiSched-GNN — implementation on the GreenDyGNN artifact

Trace-aware **optimal** rebuild scheduling: solve the cache-window problem
*exactly* offline with a dynamic program, and replace GreenDyGNN's Double-DQN
with a **no-regret** online controller. Built directly on this artifact's
sampler / cache / double-buffered prefetcher / energy harness — the only hot-path
change is an optional, default-off non-uniform-window hook in `prefetcher.py`.

> Phase 0 first. The gate (`run_gate.py`) measures the clairvoyant ceiling — the
> DP gain over the oracle-best uniform `W` — and applies the committed decision
> rule **before** any system is built (proposal §8, §13).

## What was added

```
optisched/                     pure numpy/torch, no DGL, runs offline
  calibration.py   CostModel: T_rebuild(w)=a+b·w^c, t_miss[m], α, P̄  (A3, §5.1)
  trace.py         Trace (CSR per-batch remote nodes+owners) + SyntheticTrace (A1)
  interval_cost.py σ-independent per-interval residual misses via the §6.1
                   identity  Σ miss = A − C  (+ Option-A′ full-refresh terms)
  dp_solver.py     Theorem-1 DP (+backpointers), oracle-uniform sweep (Thm 2),
                   Option-B delta-exact DP (Thm 3), brute force, per-position W*
  regime.py        RegimeLibrary (per-regime DP schedules) + FixedShare /
                   SimpleFixedShare controllers (Theorem 5 — replaces the DQN)
  gate.py          Phase-0 gate: DP vs oracle-uniform, heterogeneity, decision rule

run_gate.py        Phase-0 driver  → gain table + JSON + figures/fig_gate.pdf
dump_trace.py      replay the seeded sampler → deterministic per-epoch traces (A1)
build_library.py   traces + CostModel → online library / clairvoyant schedule JSON
calibrate.py       profiling measurements → CostModel JSON (GreenDyGNN Alg.1)
train_optisched.py trainer: offline DP schedule + fixed-share online control
prepare_optisched.sh  dump traces + build libraries for run_benchmark.sh
tests/test_optisched.py   validates every theorem the paper rests on
```

`prefetcher.py` gained `set_schedule()` / `_win_len()`; with no schedule installed
the windowing math is byte-for-byte the legacy fixed-grid behaviour (zero overhead
for RapidGNN / GreenDyGNN).

## Quick start — Phase 0 gate (no cluster)

```bash
cd code
python3 tests/test_optisched.py                       # 22 checks
python3 run_gate.py --synthetic --epochs 30 --stretch_lens 2,3,5,7,10,15
#  → per-dataset curves + results/gate_results.json + ../figures/fig_gate.pdf
```

### The headline is a curve over timescale, not a point

A single `stretch_len` (epochs per constant-congestion stretch) silently splits the
static→DP gap between temporal and novel, so the gate **sweeps** `stretch_len` and
reports the novel gap as a curve. The claim is "novel gain is **flat and >0 across
timescales**" — which pre-empts "you picked a favourable stretch_len".

Crucial design — vary timescale, fix congestion content: the schedule is a **square
wave of fixed exposure** (exactly `duty·E` congested epochs, the rest clean; that
multiset is identical at every `stretch_len`; only the on/off *block length*
changes). Prefix-cycling a multi-regime list instead changes the per-epoch σ
exposure with `stretch_len` and confounds the curve — we found and removed that.

Guards/diagnostics, all asserted in tests:
- **leakage guard** (`stretch_leakage`): every stretch is whole-epoch and
  single-σ, else flagged — a straddling/mixed stretch would let the per-stretch
  baseline average across congestion levels and *inflate* the novel gap.
- **novel flat** across timescale (the must — a rising novel curve is the
  leakage/averaging signature).
- temporal **not rising** with timescale.
- ordering static ≥ per-stretch ≥ per-epoch ≥ DP at every `stretch_len`.

Note the genuine tension: with exposure fixed, the oracle per-stretch advantage is
exposure-determined, so **temporal comes out ~flat too** — temporal-decreasing and
novel-flat are mutually exclusive under one timescale knob (both gains depend on
exposure). We prioritise novel-flat (the robustness claim) and report temporal as
stable.

**Read temporal correctly.** It is the **oracle per-stretch upper bound at fixed
exposure** — the best any *lag-free* congestion-reactive uniform controller could
do. It is **not** GreenDyGNN's realized gain: the DQN has reaction lag (needs
several boundaries of median-of-30 fetch samples to detect a shift), so it
degrades on fast churn even though the lag-free oracle does not. The oracle-vs-DQN
gap is a separate, later (favourable) measurement — temporal flatness says nothing
about it. The figure axis is labelled "exposure held fixed" so no one mistakes the
panel for a timescale-robustness claim about exposure.

**Baseline collapse, and when it reopens.** Under the square wave every stretch is
single-σ, so **per-stretch ≡ per-epoch** (the four baselines collapse to three);
the gate logs `pStr->pEp%` ≈ 0 to make this explicit. The four-baseline machinery
is retained for the general case: on **real traces**, epochs within a constant-σ
stretch can want different W (trace variation), so per-stretch and per-epoch
diverge, the leakage guard becomes a live check, and the **leak-guarded
per-stretch→DP** gap is the strict number to report (possibly below the synthetic
per-epoch→DP — the conservative, correct direction). The gate logs the σ-exposure
multiset and the `pStr->pEp%` divergence per dataset so you can see at a glance
whether real congestion is square-wave-like (trust the synthetic intuition) or has
heavy within-stretch structure (transfers weakly). The transferable claim is the
*predictor*: within-stretch W*(t) variance rank-orders the novel gap — if it still
rank-orders on real data, the per-dataset characterisation is earned.

### The gate is decomposed — read the right number

DP-gain over one-W-forever conflates two things; the gate separates them under
piecewise-stationary congestion (A7):

| baseline | what it is | who captures it |
|---|---|---|
| global-static W | one W for the whole run | — |
| **per-stretch W** | best W per constant-σ stretch | the oracle of GreenDyGNN's DQN |
| per-epoch W | best W each epoch, uniform within | a stronger uniform controller |
| **DP** | non-uniform within the epoch | OptiSched only |

- **temporal gain** = static → per-stretch — the mechanism GreenDyGNN *already*
  has (its w/o-RL ablation: 6.9–8.6 %). Not our contribution.
- **NOVEL gain** = per-stretch → DP — within-epoch non-uniformity at fixed
  congestion, which a trace-blind controller structurally cannot get. **This is
  the number the paper lives or dies on**, and the decision rule keys on it
  (STRONG ≥3 % on both larger datasets; FALLBACK if ≈0 everywhere).

Split heterogeneity matches: *across-stretch* CV of optimal W predicts the
temporal gain; *within-stretch* variance of W*(t) predicts the novel gain.

The synthetic demo already shows the trap: on ogbn-products a ~12 % gain over
static splits into ~10 % temporal (GreenDyGNN's) and only ~1.3 % novel; reddit is
the opposite (novel > temporal). Use the deployed `n_hot` (RapidGNN: 100k) on real
traces and report what the per-stretch→DP gap says — do not tune `n_hot` to inflate
it (if the interior regime needs an undeployable `n_hot`, that's a negative result
about operating range, worth reporting).

### If the novel gain is small, two contributions still survive
The no-regret **guarantee** replacing the DQN (real regardless of gain magnitude;
removes 50k training episodes + the sim-to-real gap) and **optimal** per-owner
allocation vs the fixed 60 % template (their ablation values allocation at ~3.3 %).
That's a defensible paper with the smaller "guaranteed adaptation" headline —
which is exactly what the gate exists to tell you before Phase 2.

## Full path — real traces on the cluster

**Freeze order matters.** `prepare_optisched.sh` *creates/updates* `calib.json`, so
freezing the manifest against it first would hash a stale/missing file. Use the
double-freeze: (1) pre-data code/rule freeze, (2) dump+calibrate, (3) re-freeze with
the now-final calibration, (4) run. The result is bound to the post-calibration hashes.

```bash
# 1. PRE-DATA freeze: lock the code + preregistration + ledger hashes
./freeze_repro.sh

# 2. dump deterministic traces AND calibrate (this writes/updates calib.json)
DATASET_ROOT=/path ./prepare_optisched.sh --datasets reddit --batch-sizes 2000
python3 calibrate.py --measurements meas.json --out calib.json   # at deployed n_hot=100k

# 3. POST-CALIBRATION freeze: bind provenance to the final calib.json + node IDs
./freeze_repro.sh --model calib.json

# 4. gate on real traces (result JSON also self-stamps prereg+ledger+code hashes)
python3 run_gate.py --traces traces --model calib.json \
    --datasets reddit,ogbn-products,ogbn-papers100M --n_hot 100000

# 5. if the gate verdict warrants it, run the system head-to-head vs the DQN
DATASET_ROOT=/path ./prepare_optisched.sh           # builds libraries for all configs
DATASET_ROOT=/path ./run_benchmark.sh --with-congestion \
    --methods optisched,optisched_dp,greendygnn
```

`optisched` = online fixed-share over regime experts; `optisched_dp` = clairvoyant
offline DP (the ceiling). Both read artifacts from `OPTISCHED_LIB_DIR` (default
`code/libraries/`).

## Communication-energy floor architecture (Arch doc, Theorems A–H)

The spine is reframed around a **feasible-transfer communication floor** (a lower
bound), not just "DP gain over uniform". See `OPTISCHED_LEDGER.md` for the full
exact/conditional/measured/not-claimed ledger. New modules:

- `floor.communication_floor(trace, model)` → **L_0** (Thm A): the exact
  communication-energy floor `Σ ε_init(m)⌈|U_m|/C_max⌉ + Σ κ_m d|U_m|`. Depends
  only on the trace footprints, caps, and measured coefficients — no schedule.
- `floor.schedule_transfers(...)` + `floor.near_floor(...)` → **ρ, γ_R, φ, η, n̄,
  n\*** (Thm F): the realised schedule's distance to the floor, decomposed into
  initiation and payload inflation. `ρ = E_win/L_0 = (γ_R+ηφ)/(1+η)` (exact
  identity, validated). Initiation-dominated regime flagged when `η ≪ 1`.
- `floor.lifetime_fit(...)` → Lemma D sufficient condition for `φ=1`.
- `floor.working_set_drift(w)` → **β(w)** (Arch §15), the structural predictor.
- `interval_cost.weighted_template(model, σ)` → **Thm B** weighted hot-set
  (top `f_I(u)·c(u)`); reduces to top-frequency at σ≡1.
- `dp_solver.solve_owner_decoupled(...)` → **Thm G** exact per-owner DP (removes
  owner-synchronization slack for fixed budgets); `solve_budget_templates` is the
  Object-2 static-template optimum.
- `regime.SimpleFixedShare(switch_cost=…)` → **Thm H** regret now includes the
  explicit cache-transition (switching) cost and the `2ε_mdl·M` model-error floor.

`CostModel` gained `eps_init[m]`, `kappa[m]`, `d`, `c_max[m]`; defaults are
initiation-dominated (crossover n\*≈1000 rows). Calibrate them per cluster
(`calibrate.py`); the algorithm + procedure generalize, the coefficients don't.

## Faithfulness notes (scope of the optimality claim)

* Optimality is **w.r.t. the calibrated model** (A3), the fixed per-interval cache
  rule (A5), and the window bound (A6) — not "globally optimal caching" (§6.1).
* Misses are **trace-exact** (we drop the logistic `h(W)` fit); only the rebuild
  term uses the calibrated average. Option A′ gives a *certified* upper bound that
  does not trust that average at all.
* `T_rebuild(w)=a+b·w^c` (length-only) is an unbiased average of the delta rebuild,
  so the validated ordering is **B ≤ A′** (delta ≤ full-refresh); A is the
  unbiased middle, not a per-instance bound (§5.3 — encoded in the tests).
* The online layer is **full-information** (Hedge/fixed-share, Theorem 5), legitimate
  because counterfactual per-window costs are computable from the model + observed
  σ̂ (A8); the `2·ε_mdl·T` model-error floor is reported, not hidden.
* A1 (deterministic trace) requires a **seeded** sampler — `dump_trace.py` seeds
  torch/numpy/dgl; seed the trainer identically for the clairvoyance claim to hold
  exactly. Without seeding the schedules still run and degrade gracefully.

## Complexity

Interval precompute `O(B·W_max·F̄·log n_hot)` (seconds offline, small B); the DP
itself `O(B·W_max·|T|)` (microseconds). σ-independence of the residual tables means
the whole regime library and every online counterfactual are dot products.
