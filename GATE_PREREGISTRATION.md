# OptiSched-GNN Phase-0 gate — PRE-REGISTRATION

Written **2026-06-26, before the real-trace cluster run**, while the answer is
unknown. These rules decide how the real `run_gate.py --traces` output is read.
They exist so a "mixed, graph-dependent" result — which the synthetic already
warns is the likely case (Reddit clears, Products doesn't, Papers neither) — gets
read by the rule, not argued past after the fact.

Inputs are fixed in advance: deployed `n_hot = 100000` (RapidGNN's value; a
measurement, not a knob), square-wave fixed-exposure congestion, stretch sweep
`{2,3,5,7,10,15}`, three datasets (Reddit, OGBN-Products, OGBN-Papers100M).

---

## Rule 1 — Verdict thresholds (committed before seeing the data)

A dataset **clears** iff its novel gap (leak-guarded **per-stretch → DP**) is
`novel_mean ≥ 3%` AND `novel_flat` across the sweep AND `leakage_total == 0`.

Count the datasets that clear:

| # clearing | verdict | what the paper claims |
|---|---|---|
| **≥ 2 of 3** | **STRONG** | universal: exact-optimal non-uniform scheduling beats the oracle GreenDyGNN mechanism |
| **exactly 1** | **PER-DATASET** | the offline contribution is real but **graph-dependent** — characterise *which* graphs (high within-epoch locality turnover), do **not** make a universal claim |
| **0** | **FALLBACK** | lead with the no-regret **guarantee** replacing the DQN + **optimal** per-owner allocation; the offline ceiling is an honest negative |

This rule is encoded in `gate.decision_from_curves` so the code emits the verdict;
I do not get to pick it after seeing the numbers. The "exactly 1" row is the
anticipated case and is a *legitimate paper*, not a failure — see framing note.

## Rule 2 — What "the predictor still works" requires

The predictor claim is: *within-stretch W\*(t) variance rank-orders the novel
gap.* Three datasets give exactly ONE cross-dataset ordering to check (reddit >
products > papers), and a 3-point match has a 1-in-6 chance of being spurious. So:

- A clean 3-point cross-dataset ordering is **necessary but not sufficient**.
- The predictor claim rests on the **pooled relationship across all
  `6 stretch_len × 3 datasets = 18` points** (within_stretch_het vs novel_gain),
  reported as a Spearman rank correlation (`predictor_pooled`). State the pooled
  number; do not let a lucky 3-point ordering stand in for it.
- If the pooled relationship does **not** hold, treat the per-dataset verdicts as
  unexplained and investigate before trusting any single dataset's result.

## Rule 3 — The novel gap IS per-stretch → DP, always

On real (non-square-wave) traces per-stretch and per-epoch will diverge
(`pStr->pEp% > 0`) because epochs within a constant-σ stretch want different W.
When they diverge:

- Report the **leak-guarded per-stretch → DP** gap as *the* novel gap. This is
  structurally enforced in code: `novel_gain_pct = pct(per_stretch, dp)`.
- The real per-stretch → DP gap will likely come in **below** the synthetic
  per-epoch → DP. That is the conservative, correct direction.
- **Do not** substitute `per_epoch → DP` because it looks better. `per_epoch → DP`
  on a non-square-wave trace is the leaky number this whole round eliminated.
  Pre-committed: it is reported only as the secondary `within_epoch_gain_pct`.

## Exposure / divergence diagnostic — how much to trust the transfer

The gate logs the σ-exposure multiset and `pStr->pEp%` per dataset:

- `pStr->pEp% ≈ 0` (square-wave-like real congestion) → the synthetic intuition
  transfers; the leakage guard stayed effectively clean.
- `pStr->pEp%` large (heavy within-stretch σ structure) → synthetic validation
  transfers weakly; lean on the leak-guarded per-stretch number and the pooled
  predictor, and report the divergence so readers know the regime.

## Temporal is the oracle upper bound — not GreenDyGNN

`temporalUB%` is the lag-free oracle per-stretch upper bound at fixed exposure. It
is NOT GreenDyGNN's realized gain (the DQN has reaction lag and degrades on fast
churn). The oracle-vs-DQN gap is a **separate later measurement** and is not read
from this gate.

---

## Reading commitments (added 2026-06-26, still before the run)

These bind the *reading*, the same way Rules 1-3 bind the measurement.

- **Don't anchor on the synthetic predictor (pooled 0.94 / 3pt 1.0).** Those
  numbers only show the machinery works on a generator whose turnover was built to
  rank-order. The real finding is whether the **pooled** Spearman survives on real
  traces where heterogeneity wasn't placed on purpose. A real pooled number well
  below 0.94 is **not a code failure** — it is the actual result about whether
  within-stretch W\*(t) variance predicts the novel gap in nature. Report it as-is.

- **`pStr->pEp%` magnitude tempers transfer confidence, regardless of verdict.**
  Pre-decided: *small* divergence ⇒ real congestion is square-wave-like and the
  synthetic intuition transfers; *large* divergence ⇒ heavy within-stretch
  structure, synthetic validation transfers weakly, so **even a clean clearing
  verdict is stated with that caveat attached.** The diagnostic gives the number;
  the tempering is committed now, not negotiated when it prints.

- **The verdict selects the next deferred measurement.** STRONG/PER-DATASET ⇒ the
  oracle-vs-DQN reaction-lag gap stays parked (footnote). **FALLBACK ⇒ it becomes
  load-bearing:** the paper's weight moves to "guaranteed online control replacing
  the DQN," so the realized advantage over GreenDyGNN's *lag-bearing* behavior (not
  the oracle upper bound) is the required next experiment. So the gate's verdict
  immediately implies the next step — no fresh planning round.

- **The line that costs something:** `decision_from_curves` is final, not a first
  draft to negotiate. The only time this commitment is tested is when the number is
  disappointing and a STRONG headline still feels one small reframing away. That is
  the moment the pre-registration exists for. Hold it.

### Framing note for the PER-DATASET / FALLBACK outcomes

If the verdict is PER-DATASET: the paper is "trace-driven non-uniform scheduling
helps in proportion to within-epoch locality turnover; here are the graphs where
it dominates and why," plus the guarantee and allocation contributions. That is an
honest, defensible systems paper with a characterisation headline, not a retreat.

If FALLBACK: the headline is "a no-regret online controller that matches/*beats*
the DQN with zero training and a guarantee, plus optimal allocation," and the
offline ceiling is reported as a measured negative about operating range. Also a
paper, and the gate existing is what let us know which one before Phase 2.
