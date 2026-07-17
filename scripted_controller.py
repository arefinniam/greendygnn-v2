"""Scripted-W controller for transition-cost experiments (RESEARCH_PLAN_v2 item 2).

Drop-in replacement for GreenDyGNNController in train_greendygnn.py
(--w_script): implements the observe()/decide() interface consumed by
BatchPrefetcher._run_controller_boundary, but returns a predetermined W
schedule. Used to measure the state-dependent transition cost
g(W_old, W_new, ...) with real Joules, at RANDOMIZED boundaries so the fit
sees varied cache overlaps and builder states.

Schedule clock: PLANNED batches. Each decide() advances the internal clock by
the W it returns (the trainer runs with rl_enabled, so our W is applied
verbatim). Stale-window extensions add served batches without a decision, so
the realized timeline can drift from plan by a few batches — analysis must
read decision_log / rebuild_log (ground truth), never this script.

Script grammar:
  "16@0,128@480,16@960"   explicit: W@start_planned_batch (ascending starts)
  "random:SEED:MIN:MAX"   seeded random walk: W is HELD until the dwell
                          threshold (~U[MIN,MAX] planned batches) is crossed;
                          each switch then draws from the ladder EXCLUDING the
                          current W, so every SWITCH is a real transition
                          (boundaries within a dwell keep the same W).
                          Deterministic given SEED. Because actions have
                          variable duration, a switch requested at planned
                          batch b lands at the first boundary >= b.
"""

import random

from greendygnn_agent import Decision

W_LADDER = (1, 2, 4, 8, 16, 32, 64, 128)


class ScriptedWController:
    def __init__(self, script, initial_w=16):
        self.planned_batches = 0
        self.schedule_log = []          # [(planned_batch, W), ...]
        self._explicit = None
        self._rng = None
        self._dwell = None
        self._cur_w = int(initial_w)
        self._next_switch = None

        script = script.strip()
        if script.startswith("random:"):
            parts = script.split(":")
            if len(parts) != 4:
                raise ValueError(f"bad random script {script!r} "
                                 f"(want random:SEED:MIN:MAX)")
            seed, lo, hi = int(parts[1]), int(parts[2]), int(parts[3])
            if not (0 < lo <= hi):
                raise ValueError(f"bad dwell range [{lo},{hi}]")
            self._rng = random.Random(seed)
            self._dwell = (lo, hi)
            self._cur_w = self._rng.choice(W_LADDER)
            self._next_switch = self._rng.randint(lo, hi)
        else:
            entries = []
            for tok in script.split(","):
                w, start = tok.strip().split("@")
                w, start = int(w), int(start)
                if w not in W_LADDER:
                    raise ValueError(f"W={w} not in ladder {W_LADDER}")
                entries.append((start, w))
            if not entries or entries[0][0] != 0:
                raise ValueError("explicit script must start with W@0")
            if entries != sorted(entries):
                raise ValueError("explicit script starts must be ascending")
            self._explicit = entries

    # -- BatchPrefetcher controller interface --------------------------- #
    def observe(self, owner_stats, hit_rate, step_time, base_step_time):
        """No-op: the script does not react to observations (that's the point
        — transitions must be exogenous to system state for a clean fit)."""

    def decide(self):
        if self._explicit is not None:
            w = self._cur_w = max(
                (e for e in self._explicit if e[0] <= self.planned_batches),
                key=lambda e: e[0])[1]
        else:
            if self.planned_batches >= self._next_switch:
                choices = [w for w in W_LADDER if w != self._cur_w]
                self._cur_w = self._rng.choice(choices)
                self._next_switch = self.planned_batches + \
                    self._rng.randint(*self._dwell)
            w = self._cur_w
        self.schedule_log.append((self.planned_batches, w))
        self.planned_batches += w
        return Decision(W=w, owner_budgets=None, khat={}, sigma_hat={},
                        provenance="scripted")
