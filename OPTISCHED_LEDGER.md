# OptiSched-GNN — Exact / Conditional / Measured / Not-claimed ledger

Mirrors Architecture §19. Every row names the claim, its status, and the code
object that implements or measures it. The organizing rule: **exact results are
labeled exact; conditional results state their conditions; measured quantities are
reported as measurements; open cases are named, not hidden.**

All code paths are validated in `tests/test_optisched.py` (47 checks, all passing).

## EXACT (proved; validated against brute force where applicable)

| # | Claim | Code | Test |
|---|---|---|---|
| 1 | Weighted hot-set optimality in a fixed window — top-`n_hot` by `f_I(u)·c(u)` (Thm B) | `interval_cost.weighted_template` + `precompute` | weighted ≤ freq; reduces to top-freq at σ=1 |
| 2 | Feasible-transfer communication floor `L_0` (Thm A) | `floor.communication_floor` | sandwich `ρ = E_win/L_0 ≥ 1` |
| 3 | Exact global-window DP under self-contained interval costs (Thm C) | `dp_solver.solve_optionA` | `== brute_force` (gap 6e-17) |
| 4 | Exact interval-transition DP under delta rebuilds (Thm D) | `dp_solver.solve_optionB` | `B ≤ A′` ordering |
| 5 | Exact owner-decoupled DP for fixed per-owner budgets (Thm G) | `dp_solver.solve_owner_decoupled` | per-owner `== brute_force`; total = Σ |
| 6 | Dominance over oracle-uniform `W` in the windowed class (Thm E) | `dp_solver.oracle_uniform` | `DP ≤ oracle_uniform` |

## CONDITIONAL (holds under stated, checkable conditions)

| # | Claim | Condition | Code |
|---|---|---|---|
| 1 | `ρ = γ_R + O(η·φ)` | initiation-dominated regime, `η ≪ 1` | `floor.near_floor.initiation_dominated` |
| 2 | `γ_R ≤ P−1` | #global windows ≤ capacity-forced delivery count `K ≤ R_0` | `floor` R_win vs R_0 (per-trace check) |
| 3 | `γ_R → 1` | owner-decoupled + sufficient fixed budgets + lifetime-fit/block-locality | `solve_owner_decoupled` + `floor.lifetime_fit` |
| 4 | `ρ → 1` | initiation dominance AND initiation inflation → 1 | `near_floor` (reported, not assumed) |
| 5 | Static budget-template optimum | exact **only within** the template set (Object 2) | `dp_solver.solve_budget_templates` |
| 6 | `φ = 1` (payload-floor attainment, Lemma D) | contiguous lifetimes AND live-set ≤ capacity | `floor.lifetime_fit` |
| 7 | Novel scheduling gain flat across timescale | fixed congestion exposure (square wave) | `gate` sweep + `GATE_PREREGISTRATION.md` |

## MEASURED (reported as numbers, never assumed)

| Quantity | Symbol | Code |
|---|---|---|
| Max stable bulk-transfer size | `C_max[m]` | `CostModel.c_max` (calibrated) |
| Payload-to-initiation ratio at floor | `η` | `floor.Floor.eta` |
| Realised / floor avg payload per transfer | `n̄`, `n̄_0` | `near_floor.nbar`, `Floor.nbar0` |
| Payload crossover size | `n* = 1/θ` | `floor.Floor.n_star_mean`, `CostModel.n_star()` |
| Payload inflation | `φ = Q_win/Q_0` | `near_floor.phi` |
| Initiation inflation | `γ_R = R_win/R_0` | `near_floor.gamma_R` |
| Distance to floor | `ρ = E_win/L_0` | `near_floor.rho` |
| Working-set drift | `β(w)` | `floor.working_set_drift` |
| β ↔ ρ, γ_R, φ, scheduling-gain correlation | — | gate pooled predictor + β |
| Online reaction-lag gap vs GreenDyGNN | — | **deferred experiment** (see ledger "not yet run") |
| Model error on held-out windows / congestion | `ε_mdl` | `CostModel.eps_mdl` (calibration output) |

## NOT CLAIMED (explicitly disavowed)

1. Unrestricted optimal caching is solved. (We solve the windowed schedule for the
   measured instance; the floor is conditional on fixed sampler/partition/feature-
   rep/transport + measured caps/coefficients.)
2. `ρ = 1` unconditionally.
3. One calibration transfers unchanged across machines. (Algorithm + calibration
   *procedure* generalize; fitted coefficients are cluster-specific — §18.)
4. GreenDyGNN ignores the trace entirely. (It is trace-aware in cache construction
   but not trace-optimizing in window selection — §17, the fair statement.)
5. No method outside the model can reduce energy. (Other techniques act by changing
   the floor's inputs `|U_m|`, `C_max`, `ε_init`, `κ_m d`; scheduling sets how close
   we get to the floor — §5.)
6. Fixed per-owner budget DP solves fully time-varying shared-budget allocation.
   (Object 3 is a named open problem — §14.)

## Not yet run (requires the cluster)

- Real-trace gate at deployed `n_hot=100000` → per-dataset novel curves, pooled
  predictor, exposure/divergence (read against `GATE_PREREGISTRATION.md`).
- Floor calibration: measure `ε_init(m), κ_m, d, C_max[m]`, rebuild table,
  idle/active power, congestion-estimator accuracy, held-out model error (§18).
- If the gate verdict is FALLBACK: the oracle-vs-DQN reaction-lag gap becomes the
  load-bearing experiment (Conditional/Measured row above).

## Final theorem package (Architecture §20) → code map

| Theorem | Statement | Code |
|---|---|---|
| A feasible-transfer floor | `E_c(Π) ≥ L_0` | `floor.communication_floor` |
| B weighted hot-set | optimal cache = top `f_I(u)c(u)` | `interval_cost.weighted_template` |
| C exact windowed DP | exact under self-contained costs | `dp_solver.solve_optionA` |
| D exact delta-transition DP | exact under delta rebuilds | `dp_solver.solve_optionB` |
| E dominance over oracle-uniform | `OPT_win ≤ min_W E(uniform-W)` | `dp_solver.oracle_uniform` |
| F near-floor decomposition | `ρ = (γ_R+ηφ)/(1+η)` | `floor.near_floor` |
| G owner-decoupled exact DP | `Σ_m OPT_m`; `γ_R→1` under conditions | `dp_solver.solve_owner_decoupled` |
| H online tracking | fixed-share + switch cost + ε_mdl floor | `regime.SimpleFixedShare` |
