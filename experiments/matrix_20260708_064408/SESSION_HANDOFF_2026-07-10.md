# GreenDyGNN v2 — Session Handoff (2026-07-10)

This document is written for an agent starting a fresh session with no prior context.
Read it top to bottom before touching anything. It is accurate as of 2026-07-10 ~16:50 UTC.

Deeper background lives in the auto-memory files (load these too):
`greendygnn-v2-campaign`, `optisched-gnn`, `optisched-p1-joules`, `gnn-p100-cluster`.
This handoff supersedes them for the *current run state* but not for the multi-week history.

---

## 1. What this project is

GreenDyGNN is a distributed-GNN energy-efficiency system (windowed feature cache + an
online controller) being rebuilt for a **journal resubmission** after the SC26 version was
rejected (synthesized figures, no real simulator, marginal 4% gain, no error bars).

The v2 system has two levers, both driven by one online controller at each cache-rebuild
boundary:
- **RL window selection (W)** — a frozen Double-DQN (`dqn_v2_real.pt`) picks the window
  length W ∈ {4,8,16,32,64,128}. Trained offline in a real-calibrated simulator.
- **Analytic per-owner cache allocation** — tilts the cache toward "expensive" owners,
  keyed on `sigma_hat` = per-owner mean-RTT ratio vs that owner's *own* warm-up baseline
  (self-normalized, so a static per-owner cost difference is absorbed and only real
  *degradation* tilts the cache).

The paper's evidence is a **6×2×6×3 matrix**: 6 methods × 2 datasets × 6 congestion
conditions × 3 seeds = 216 real-Joule distributed training runs.

**Methods (the 6):**
| method | W control | allocation | role |
|---|---|---|---|
| `default_dgl` | — | no cache at all | floor / worst case |
| `rapidgnn_epoch` | epoch-level cache | uniform | prior-work baseline |
| `static_w16` | fixed W=16 | uniform | strong cache baseline |
| `greendygnn_v2` | **RL** | **on** | the full proposed system |
| `v2_no_rl` | fixed W=16 | **on** | ablation: allocation only |
| `v2_uniform_alloc` | **RL** | uniform | ablation: RL-W only |

**Conditions (the 6):** see §7. `clean`, `c1_200/100/50` (steady bandwidth throttle),
`c2_duty` (organic iperf3), `c3_sq200` (square-wave throttle).

---

## 2. Where everything lives

**Cluster** (`gnn-p100-cluster` memory has full detail): 4 Chameleon bare-metal nodes,
user `cc`, key `~/.ssh/newtacckey.pem`. Public IPs from local WSL:
gnn1 `129.114.108.218`, gnn2 `.186`, gnn3 `.115`, gnn4 `.101`.
Private (node-to-node only) `10.52.x`: gnn1 `10.52.2.119`, gnn2 `.3.217`, gnn3 `.3.123`,
gnn4 `.3.89`. gnn4 = **owner3** = the congestion victim.

- **Deployed code (all 4 nodes, md5-identical):** `~/gdy2/` on each node.
  The trainers, `prefetcher.py`, `alloc.py`, `optisched/`, `run_matrix.sh`,
  `congestion2.py`, `checkpoints/dqn_v2_real.pt`.
- **Local working copy (source of truth for edits):** `~/greendygnn_work/code/`
  on the WSL box (this machine). Edit here, then `scp` to all 4 nodes.
- **Measurement tooling:** `~/optisched_sys/aggregate_p1.py` on gnn1 (the aggregator —
  NOT in `~/gdy2/`, common footgun).
- **Results:** `~/matrix_results/matrix_20260708_064408/` on gnn1. Per-combo run dirs at
  `runs/<dataset>__<cond>__<method>__seed<N>/`, each holding 4 part profiles
  + `p1_aggregate.json` (the per-epoch system-energy summary) + `run.log` + `cong_journal.jsonl`.
- **Env on cluster:** MUST `source ~/dt-venv/bin/activate && source ~/dt-env.sh` before
  any DGL work (the second sets `LD_LIBRARY_PATH` for `libcusparse.so.11`; without it
  `import dgl` fails).

---

## 3. CURRENT STATE — what is running right now

- **Main matrix: DONE.** 204/216 combos have valid `p1_aggregate.json`. Driver exited
  cleanly (`matrix complete` in `~/gdy2/matrix_full.log`). Zero unresolved failures.
- **Patch-up pass: RUNNING.** The 12 `default_dgl` × {c1_100, c1_50} × 3 seeds runs were
  deferred during the main matrix because they blow the default 3600s timeout (no cache
  under heavy throttle → very slow epochs). Relaunched 2026-07-10 ~16:34 UTC with
  `--timeout 9000`, driver **pid 2377260** on gnn1, log `~/gdy2/matrix_patchup.log`.
  As of this writing it is on `reddit c1_100 default_dgl`. Expect ~3–4 h to clear all 12.
  When it finishes the matrix is 216/216. These are just the no-cache floor numbers under
  heavy congestion; they do not change any conclusion.
- **Cluster hygiene:** clean between the two phases (verified 0 python procs, `tc` clean on
  all nodes) except the patch-up driver's own startup found 1 leftover proc each on
  gnn3/gnn4 and killed them — normal.

To resume/re-run either phase: same `run_matrix.sh` invocation with the same `--out` dir
skips any combo that already has a `p1_aggregate.json` (resume is by file existence).

---

## 4. The bug fixed this session (allocation contract bug) — DONE, verified

At 45/216 runs the v2 methods were catastrophic even on `clean` (e.g. greendygnn_v2 clean
5659 J/ep vs static 1346; cache hit collapsing 82%→~0.02% at the data-driven warmup exit
~epoch 8). Root cause was **two stacked I2 controller↔prefetcher contract bugs** in
`prefetcher.py`:

1. When the controller sent `owner_budgets=None` (meaning "uniform"), the prefetcher
   tilted allocation from the raw `khat` cross-owner ratio anyway (fetch-size-confounded,
   read 25.9× on a *clean* run) → hit-rate collapse.
2. That collapse ballooned fetch sizes → `sigma_hat` spuriously inflated → controller then
   sent `owner_budgets` = allocation *weights* (capped 1–8), which the prefetcher consumed
   as absolute *row budgets* (`take=min(int(budget),cnt)`) → built an ~8-row cache. Death spiral.

**Fix (in `~/greendygnn_work/code/prefetcher.py`, deployed all 4 nodes):**
`Decision.owner_budgets` (weights) are routed through the analytic allocator via
`controller_khat`; `owner_budgets=None` → uniform (raw khat NEVER steers); the
row-budget path was deleted entirely. Added regression test
`test_alloc_weights_routed_and_cache_stays_full` (also exposed that the pipeline test
harness had **no partition book**, so every owner-aware path was silently untested — the
blind spot that let this ship). Tests: **100/100 local, 99/99 cluster** (gnn1 needed
`pip install pytest`).

Verified fixed: post-fix clean greendygnn_v2 holds hit ~70% (not 0.02%), and the entire
re-run matrix produced sane numbers. **This bug is closed.**

Note on the ~15 `FAIL(exit=255)` rows you'll see referenced in logs: those were a
*separate*, benign DGL distributed-**shutdown** race (SIGABRT on the exit barrier, ~10%
flake across ALL methods incl. plain baselines). Training completed fully in every case;
all 15 were salvaged by re-running `aggregate_p1.py` against their finished profiles and
marked `OK(salvaged-shutdown-crash)`. Not a correctness issue.

---

## 5. RESULTS — full table and honest interpretation

Units: `J/ep` = system Joules per steady-state epoch (CPU RAPL + GPU NVML, summed over 4
parts, median-of-positive-deltas, warmup=5). `hit%` = remote cache hit rate. Lower J/ep
is better. Accuracy is ~0.917 (reddit) / ~0.860 (products) for ALL methods — nobody trades
accuracy, so energy is the whole story. n=3 seeds each.

### The headline the data actually supports (READ THIS — it is not the paper's headline)

**The v2 learned controller does NOT currently beat the static baseline. Under steady
bandwidth congestion (c1_*, the primary condition) it is dramatically WORSE.** The best
v2 variant is `v2_no_rl` (fixed W16 + allocation), which roughly *ties* `static_w16` and
only edges ahead under *variable* congestion.

Two distinct problems, both real:

**(A) The RL window policy inflates energy under bandwidth throttle.** `greendygnn_v2` and
`v2_uniform_alloc` (the two RL-W methods) pick large W (W=128 observed live) under c1.
Large W = bigger bulk rebuild transfers that pay the throttled bandwidth → longer epochs →
more integrated energy. Examples (reddit):
- `c1_200`: greendygnn_v2 **12219** vs static_w16 **3751**, v2_no_rl **4030** → RL is ~3× worse.
- `c1_100`: greendygnn_v2 **13027**, v2_uniform_alloc **29293** vs static **6595**, v2_no_rl **7752**.
- `clean`: greendygnn_v2 **1737** vs static **1346** (~29% big-W tax even with no congestion).

This is a **sim-to-real gap in the deployed checkpoint**. The offline sim said the DQN wins
by *not shrinking* W on bandwidth (correct — small W is catastrophic there); but live it
*over-shoots to very large W*, which is also wrong. The calibrated finding was that reddit
bandwidth congestion wants **moderate** W (~16). The frozen policy is mis-mapping real
observations to actions.

**(B) The allocation lever does not reproduce its P1 win — and self-normalization explains
why.** `v2_no_rl` (allocation, fixed W) is ~7–18% *worse* than `static_w16` under c1, not
better:
- reddit c1_200: v2_no_rl 4030 vs static 3751 (+7%); c1_100 7752 vs 6595 (+18%); c1_50 14400 vs 12254 (+18%).

Mechanism: **c1 applies the throttle from run start, so `sigma_hat`'s self-normalized
per-owner baseline (frozen at warm-up) already includes the throttle → the controller
cannot see the asymmetry → no beneficial tilt, and the small tilt it does apply is
counterproductive.** The 2026-07-07 self-normalization redesign (which correctly fixed the
fetch-size confound) has this side effect: **allocation is blind to congestion that is
present at baseline time.** It can only react to congestion that *onsets mid-run* — and
that is exactly where v2_no_rl shows its only wins:
- reddit `c2_duty`: v2_no_rl **1497** is the BEST (static 1600, rapidgnn 1709).
- products `c2_duty`/`c3_sq200`/`clean`: greendygnn_v2 & v2_no_rl beat default_dgl and
  rapidgnn, roughly tie static (all ~2200 J/ep on products clean).

### Full table (as of 204/216; default_dgl c1_100/c1_50 still filling)

```
ds             cond     method              J/ep   hit%
reddit         clean    default_dgl         4112    0.0
reddit         clean    rapidgnn_epoch      1517   83.0
reddit         clean    static_w16          1346   81.6   <- baseline to beat
reddit         clean    greendygnn_v2       1737   70.0   (worse: big-W tax)
reddit         clean    v2_no_rl            1347   81.2   (ties static)
reddit         clean    v2_uniform_alloc    1801   70.7
reddit         c1_200   default_dgl        37658    0.0
reddit         c1_200   rapidgnn_epoch      5046   56.7
reddit         c1_200   static_w16          3751   59.1   <- best
reddit         c1_200   greendygnn_v2      12219   52.2   (3x worse)
reddit         c1_200   v2_no_rl            4030   60.4
reddit         c1_200   v2_uniform_alloc   11835   51.7
reddit         c1_100   static_w16          6595   52.9   <- best
reddit         c1_100   greendygnn_v2      13027   42.6
reddit         c1_100   v2_no_rl            7752   53.9
reddit         c1_100   v2_uniform_alloc   29293   53.8   (worst)
reddit         c1_50    static_w16         12254   47.5   <- best
reddit         c1_50    v2_no_rl           14400   47.1
reddit         c1_50    greendygnn_v2      25413   46.2
reddit         c2_duty  v2_no_rl            1497   79.6   <- v2 WINS here
reddit         c2_duty  static_w16          1600   77.5
reddit         c2_duty  rapidgnn_epoch      1709   79.3
reddit         c2_duty  greendygnn_v2       2060   67.9
reddit         c3_sq200 static_w16          1350   77.6   <- best
reddit         c3_sq200 v2_no_rl            1392   75.6   (near tie)
reddit         c3_sq200 greendygnn_v2       2049   63.5
products       clean    static_w16          2192   46.5
products       clean    v2_no_rl            2201   46.3   (ties static)
products       clean    greendygnn_v2       2203   45.7   (ties static)
products       clean    default_dgl         2659    0.0
products       clean    rapidgnn_epoch      2695   42.4   (v2 beats rapid)
products       c1_200   static_w16          9692   29.7   <- best
products       c1_200   v2_no_rl            9713   29.7   (ties)
products       c1_200   greendygnn_v2      12306   34.4
products       c1_50    static_w16         34364   30.2   <- best
products       c1_50    v2_no_rl           34515   29.8   (ties)
products       c1_50    greendygnn_v2      45582   37.3
products       c2_duty  static_w16          2390   44.4
products       c2_duty  v2_no_rl            2423   44.0
products       c2_duty  greendygnn_v2       2533   44.0
products       c3_sq200 v2_no_rl            2207   42.4   <- best (edges static 2242)
products       c3_sq200 static_w16          2242   42.4
products       c3_sq200 greendygnn_v2       2384   50.6
```
(Regenerate anytime: `scp` `/home/arefin/.claude/jobs/*/tmp/early_table.py` isn't durable;
the script is 20 lines — it globs `runs/*/p1_aggregate.json`, groups by combo, means the
seeds. Rewrite it if lost. It skips `{"deferred":...}` marker files.)

### Summary read for the paper
- **default_dgl (no cache) is crushed** everywhere by caching (the "caching saves energy"
  story is intact and strong: reddit clean 4112→1346, ~67%).
- **static_w16 (fixed W16 + uniform) is the method to beat, and mostly nobody beats it.**
- The **proposed learned system (greendygnn_v2) currently LOSES** to its own static
  ablation, badly under c1. As-is, this does not support a "learned controller wins" paper.
- The **only place v2 wins is variable/onset congestion (c2_duty)** via allocation, and
  only by a few %.

This is the central thing the next session must grapple with. Do not write the paper around
a win that the data does not show. Options in §6.

---

## 6. What the next agent should do (in priority order)

1. **Let the patch-up pass finish** (pid 2377260) → confirm 216/216, then re-generate the
   full table. Nothing else blocks on it.
2. **Diagnose the RL big-W behavior (problem A).** Pull the decision journals
   (`runs/<...>/*part0_profile.json` → `decisions[]`) for greendygnn_v2 under c1_* and read
   the `W` and `provenance` fields. Confirm it's choosing W=128 and check the state vector /
   observation-compression mapping in `greendygnn_agent.py` against what the sim trained on.
   Likely culprit: the live observation (mean-RTT ratio / step-time ratio) maps to a region
   of state space where the frozen Q-net prefers max-W, but real energy there is bad.
   Candidate fixes: retrain/re-calibrate the checkpoint against the real c1 energy; or add
   an energy-aware guard; or restrict the action set. **Do NOT silently swap in a heuristic
   and call it the DQN.**
3. **Decide the allocation story (problem B).** The self-normalized `sigma_hat` is blind to
   steady (baseline-present) congestion by design. Either (a) reframe the paper's allocation
   claim around *onset/variable* congestion only (c2_duty is the honest win), or (b) add a
   path that measures per-owner cost during a clean pre-warmup and sets κ explicitly (this
   is what the P1 campaign did to get its 21–44% win — see `optisched-p1-joules` memory,
   `het3.sh`/`sweep_het.sh`). Note the P1 win used κ *measured under the throttle*, which
   the live self-normalized signal deliberately cannot replicate.
4. **Only after 2–3:** decide whether the honest paper is "floor-guided allocation under
   *variable* congestion + strong static caching," which is what the data supports, vs the
   original "learned two-lever controller wins," which it does not. The `optisched-gnn`
   memory already contains a fully-worked FALLBACK framing (allocation-centric) if the
   learned-controller headline can't be rescued.

Do not add architecture. Both remaining problems are calibration/eval, not design.

---

## 7. Congestion mechanisms (what "condition" means)

All shaping is `tc` on `eno1` (private NIC), **subnet-broad, never port-filtered** — this is
critical: an earlier campaign wasted hours with port-30050-scoped `netem` that never hit
DGL's feature traffic (rides ephemeral ports; tcpdump-proven). `congestion2.py` handles all
of it, with a canary/verify before and a teardown-verify after every run.

| cond | mechanism | detail |
|---|---|---|
| `clean` | none | verifies tc clean, that's all |
| `c1_200/100/50` | `tbf` steady | bandwidth cap (200/100/50 mbit) on gnn4 egress, whole run. Deterministic severity dial. **Applied from run start** (see problem B). |
| `c2_duty` | organic `iperf3` | all 3 peers → gnn4 incast, 8 streams, 30s on / 30s off. Real competing traffic. **Most realistic + most variable.** |
| `c3_sq200` | `tbf` square-wave | 200mbit cap toggled on/off, 120s period 50% duty, wall-clock scheduled (deliberately NOT epoch-synced). |

Realism/variability ranking: **c2_duty > c3_sq200 > c1**. c1 is the controlled backbone,
c2 is the real-world stressor.

---

## 8. Operational gotchas (each of these bit us this campaign)

- **`pkill -f` with a pattern containing `train_` or `run_matrix` KILLS YOUR OWN SSH
  SESSION** — the remote command's own argv matches the pattern (exit 255, session dies).
  Use `pgrep -f "^python3"` to count, target explicit PIDs to kill, or match the exact
  binary. Verify with `ps aux | grep run_matrix.sh | grep -v grep`.
- **The driver's TERM trap runs teardown but bash defers the signal until the current child
  (`launch.py`) exits.** To stop cleanly: `kill -TERM <driver_pid>`, then if it survives
  after the child, `kill -9`. SIGKILL skips teardown → **manually verify `tc qdisc show dev
  eno1` is clean on all 4 nodes afterward.**
- **`aggregate_p1.py` is at `~/optisched_sys/` on gnn1, not `~/gdy2/`.**
- **`--label` must match the profile filename prefix** or the aggregator finds 0 profiles
  and under-counts to zero. Real prefixes: greendygnn_v2→`greendygnn`,
  v2_no_rl→`greendygnn_no_rl`, v2_uniform_alloc→`greendygnn_uniform_alloc`,
  static_w16→`static_w16` (train_rapidgnn under the hood), rapidgnn_epoch→`rapidgnn_epoch`.
  Check `ls runs/<combo>/*profile*` if unsure.
- **Salvaging a shutdown-crashed run:** if `run.log` shows `Part 0 Ep29:` (training
  finished) but exit was 255, just run `aggregate_p1.py --run_dir <rd> --label <prefix>
  --warmup 5`; the 4 part profiles are already on disk. Then mark the tsv row OK so resume
  skips it.
- **Local WSL sleep kills nohup'd cluster-driving chains** between sessions (the ssh dies).
  Long runs are driven *on gnn1* via `nohup bash run_matrix.sh ... &`, not from WSL. Reconnect
  and check `ps` / logs; the run itself survives on the cluster.
- **RAPL chmod resets on node reboot;** `run_matrix.sh` re-applies a readable barrier each
  run, so normally fine.
- **Editing code:** edit `~/greendygnn_work/code/` locally, run `pytest tests/`, then `scp
  prefetcher.py`/etc to `~/gdy2/` on ALL 4 nodes and `md5sum` to confirm identical. Tests
  moved to `tests/` subdir on nodes.

---

## 9. One-liner status commands

```bash
KEY=~/.ssh/newtacckey.pem; G1=cc@129.114.108.218; M=~/matrix_results/matrix_20260708_064408
# is anything running?
ssh -i $KEY $G1 'ps aux | grep run_matrix.sh | grep -v grep; echo ---; pgrep -cf "^python3"'
# how many valid results?
ssh -i $KEY $G1 "ls $M/runs/*/p1_aggregate.json | xargs grep -l sys_total | wc -l"  # want 216
# any non-OK rows?
ssh -i $KEY $G1 "awk -F'\t' '\$5!~/^OK/' $M/matrix.tsv"
# tc clean on all nodes? (should all say 'qdisc mq 0: root')
ssh -i $KEY $G1 'tc qdisc show dev eno1|head -1; for h in 10.52.3.217 10.52.3.123 10.52.3.89; do ssh -o StrictHostKeyChecking=no $h "tc qdisc show dev eno1|head -1"; done'
```
