#!/usr/bin/env python3
"""Congestion regimes, the DP-optimal expert library, and the no-regret online
controller (proposal 6.5 / Theorem 5, Algorithm 7.3).

The online side replaces GreenDyGNN's Double-DQN.  Offline we precompute a small
library of Theorem-1 DP schedules, one per congestion regime sigma^(r).  At
runtime, at each rebuild boundary, a fixed-share (Herbster-Warmuth) controller
plays a distribution over these experts, commits the chosen expert's window, then
-- because counterfactuals are free (A8, full information) -- scores *every*
expert's action under the observed congestion sigma_hat and does a Hedge update
plus a fixed-share mix.  This carries the Theorem-5 regret guarantee against the
best regime-switching schedule in hindsight, with no training and no sim-to-real
gap.

Key efficiency point: residual misses are sigma-independent, so the whole library
is built from ONE `interval_cost.precompute`; per-regime work is just a DP and a
dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .calibration import CostModel
from .interval_cost import IntervalCosts, precompute
from .trace import Trace
from . import dp_solver as DP


# --------------------------------------------------------------------- regimes
def default_regimes(num_partitions: int, local_rank: int,
                    severities=(0.5, 1.0, 2.0)) -> List[Tuple[str, np.ndarray]]:
    """A compact regime set: clean, each owner congested at several severities,
    and a couple of multi-owner combinations (GreenDyGNN archetypes x severity).

    sigma_m is a multiplier on owner m's miss latency (1.0 = clean).  Maps to
    GreenDyGNN's injected delays via sigma_m = 1 + delay/base.
    """
    remote = [p for p in range(num_partitions) if p != local_rank]
    regimes: List[Tuple[str, np.ndarray]] = [("clean", np.ones(num_partitions))]
    for p in remote:
        for s in severities:
            sg = np.ones(num_partitions)
            sg[p] = 1.0 + s
            regimes.append((f"owner{p}+{s}", sg))
    if len(remote) >= 2:
        sg = np.ones(num_partitions)
        sg[remote[0]] = 2.0
        sg[remote[1]] = 1.5
        regimes.append(("two-owner", sg))
        sg = np.ones(num_partitions)
        for p in remote:
            sg[p] = 1.5
        regimes.append(("all-owner", sg))
    return regimes


# --------------------------------------------------------------------- library
@dataclass
class RegimeLibrary:
    """DP-optimal schedules + cost matrices, one per regime."""

    model: CostModel
    ic: IntervalCosts
    names: List[str]
    sigmas: List[np.ndarray]
    schedules: List[DP.Schedule]
    cost_mats: List[np.ndarray]          # [N][B, W_max] cost under each regime
    best_t: List[np.ndarray]
    # per-regime, per-position recommended (window length, template):
    rec_w: np.ndarray                    # int[N, B]
    rec_t: np.ndarray                    # int[N, B]

    @classmethod
    def build(cls, trace: Trace, model: CostModel, n_hot: int,
              regimes: Optional[List[Tuple[str, np.ndarray]]] = None,
              templates=None, w_max: Optional[int] = None,
              ic: Optional[IntervalCosts] = None) -> "RegimeLibrary":
        W_max = int(w_max or model.w_max)
        if regimes is None:
            regimes = default_regimes(trace.num_partitions, trace.local_rank)
        if ic is None:
            ic = precompute(trace, model, n_hot, w_max=W_max, templates=templates)
        names = [n for n, _ in regimes]
        sigmas = [np.asarray(s, dtype=np.float64) for _, s in regimes]

        schedules, cost_mats, best_ts = [], [], []
        N, B = len(sigmas), ic.B
        rec_w = np.ones((N, B), dtype=np.int64)
        rec_t = np.zeros((N, B), dtype=np.int64)
        for r, sg in enumerate(sigmas):
            cost, bt = ic.cost_matrix(model, sg, variant="A")
            sched = DP.solve_optionA(cost, bt, w_max=W_max)
            schedules.append(sched)
            cost_mats.append(cost)
            best_ts.append(bt)
            # per-position recommendation = amortised-optimal window for this regime
            for i in range(B):
                best, bw = np.inf, 1
                for w in range(1, min(W_max, B - i) + 1):
                    c = cost[i, w - 1]
                    if np.isfinite(c) and c / w < best:
                        best, bw = c / w, w
                rec_w[r, i] = bw
                rec_t[r, i] = int(bt[i, bw - 1])
        return cls(model, ic, names, sigmas, schedules, cost_mats, best_ts,
                   rec_w, rec_t)

    def controller(self, **kw) -> "FixedShareController":
        return FixedShareController(self, **kw)


# ------------------------------------------------------------------- controller
class FixedShareController:
    """Hedge + fixed-share over the regime experts (Theorem 5).

    Usage per epoch / boundary stream (mirrors Algorithm 7.3):

        ctl = library.controller()
        pos = 0
        while pos < B:
            w, t = ctl.select(pos)            # commit next window (length, template)
            ...run window, observe realized congestion -> sigma_hat...
            ctl.update(pos, sigma_hat)        # full-info Hedge + fixed-share
            pos += w

    Losses are the calibrated per-window costs (A8); they are normalised to [0,1]
    by a running maximum so the Theorem-5 constants apply.  `regret_so_far`
    reports realized regret against the single best expert in hindsight.
    """

    def __init__(self, library: RegimeLibrary, eta: Optional[float] = None,
                 alpha_fs: Optional[float] = None, sample: bool = False,
                 seed: int = 0):
        self.lib = library
        self.N = len(library.sigmas)
        self.B = library.ic.B
        self.w_max = library.ic.W_max
        self.sample = sample
        self.rng = np.random.default_rng(seed)
        # default tuning (Theorem 5): eta ~ sqrt(8 ln N / T-ish), alpha_fs ~ 1/T.
        T = max(1, self.B)
        self.eta = float(eta if eta is not None else np.sqrt(8.0 * np.log(max(2, self.N)) / T))
        self.alpha_fs = float(alpha_fs if alpha_fs is not None else 1.0 / T)
        self.w = np.ones(self.N) / self.N
        self._played: Optional[int] = None
        self._loss_scale = 1e-9
        # bookkeeping for honest regret reporting
        self.cum_loss_played = 0.0
        self.cum_loss_expert = np.zeros(self.N)
        self.rounds = 0

    def select(self, pos: int) -> Tuple[int, int]:
        """Pick an expert and return its recommended (window_length, template)."""
        if self.sample:
            i = int(self.rng.choice(self.N, p=self.w))
        else:
            i = int(np.argmax(self.w))
        self._played = i
        w = int(self.lib.rec_w[i, pos])
        w = max(1, min(w, self.w_max, self.B - pos))
        t = int(self.lib.rec_t[i, pos])
        return w, t

    def update(self, pos: int, sigma_hat: np.ndarray,
               observed_cost: Optional[np.ndarray] = None) -> None:
        """Full-information Hedge update + fixed-share mixing.

        `observed_cost` (cost matrix under sigma_hat) may be supplied to avoid a
        recompute; otherwise it is derived from the calibrated model and sigma_hat.
        """
        if observed_cost is None:
            observed_cost, _ = self.lib.ic.cost_matrix(self.lib.model, sigma_hat,
                                                       variant="A")
        # each expert's loss = realized cost of the window it would have placed here
        losses = np.empty(self.N)
        for i in range(self.N):
            w = int(self.lib.rec_w[i, pos])
            w = max(1, min(w, self.w_max, self.B - pos))
            c = observed_cost[pos, w - 1]
            losses[i] = c if np.isfinite(c) else observed_cost[pos, 0]
        self.update_with_losses(losses)

    def update_with_losses(self, losses: np.ndarray) -> None:
        """Hedge + fixed-share on an explicit full-information loss vector.

        Used for coarser (e.g. per-epoch) control cadences where the caller
        supplies each expert's counterfactual loss directly (A8).  Carries the
        same Theorem-5 guarantee at that cadence.
        """
        losses = np.asarray(losses, dtype=np.float64).reshape(-1)
        self._loss_scale = max(self._loss_scale, float(np.max(losses)))
        norm = losses / self._loss_scale                      # scale to [0,1]
        # Hedge multiplicative update
        self.w *= np.exp(-self.eta * norm)
        self.w /= self.w.sum()
        # fixed-share mix (Herbster-Warmuth)
        self.w = (1.0 - self.alpha_fs) * self.w + self.alpha_fs / self.N
        # regret bookkeeping (raw cost units)
        if self._played is not None:
            self.cum_loss_played += float(losses[self._played])
        self.cum_loss_expert += losses
        self.rounds += 1

    @property
    def regret_so_far(self) -> float:
        """Realized regret vs the single best expert in hindsight (raw cost)."""
        if self.rounds == 0:
            return 0.0
        return float(self.cum_loss_played - self.cum_loss_expert.min())

    def stats(self) -> dict:
        return {
            "experts": self.N,
            "eta": round(self.eta, 5),
            "alpha_fs": round(self.alpha_fs, 6),
            "rounds": self.rounds,
            "regret": round(self.regret_so_far, 6),
            "weights_top": int(np.argmax(self.w)),
            "model_error_floor": round(2 * self.lib.model.eps_mdl, 4),
        }


class SimpleFixedShare:
    """Library-agnostic Hedge + fixed-share over N experts (Theorem 5).

    Used by the trainer at a per-epoch (or per-boundary) cadence where the caller
    already knows each expert's action and supplies full-information loss vectors
    (A8).  Identical update to FixedShareController, without the trace coupling.
    """

    def __init__(self, n_experts: int, horizon: int = 30,
                 eta: Optional[float] = None, alpha_fs: Optional[float] = None,
                 eps_mdl: float = 0.028, switch_cost: float = 0.0, seed: int = 0):
        self.N = int(n_experts)
        T = max(1, int(horizon))
        self.eta = float(eta if eta is not None
                         else np.sqrt(8.0 * np.log(max(2, self.N)) / T))
        self.alpha_fs = float(alpha_fs if alpha_fs is not None else 1.0 / T)
        self.eps_mdl = eps_mdl
        # Explicit cache-transition (switching) cost charged when the played
        # expert changes between rounds (Architecture §16, Theorem H): switching
        # regime schedules may force a cache realignment, so it cannot be free.
        self.switch_cost = float(switch_cost)
        self.w = np.ones(self.N) / self.N
        self.rng = np.random.default_rng(seed)
        self._played = None
        self._last_played = None
        self._scale = 1e-9
        self.cum_played = 0.0
        self.cum_expert = np.zeros(self.N)
        self.cum_switch = 0.0
        self.switches = 0
        self.rounds = 0

    def select(self, sample: bool = False) -> int:
        i = int(self.rng.choice(self.N, p=self.w)) if sample else int(np.argmax(self.w))
        if self._last_played is not None and i != self._last_played:
            self.switches += 1
            self.cum_switch += self.switch_cost
        self._played = i
        self._last_played = i
        return i

    def update_with_losses(self, losses: np.ndarray) -> None:
        losses = np.asarray(losses, dtype=np.float64).reshape(-1)
        self._scale = max(self._scale, float(np.max(losses)))
        norm = losses / self._scale
        self.w *= np.exp(-self.eta * norm)
        self.w /= self.w.sum()
        self.w = (1.0 - self.alpha_fs) * self.w + self.alpha_fs / self.N
        if self._played is not None:
            self.cum_played += float(losses[self._played])
        self.cum_expert += losses
        self.rounds += 1

    @property
    def regret_so_far(self) -> float:
        """Realised regret vs the best fixed expert, INCLUDING switching cost
        (Theorem H): Σ ℓ(played) + C_switch,total − min_i Σ ℓ(i)."""
        if self.rounds == 0:
            return 0.0
        return float(self.cum_played + self.cum_switch - self.cum_expert.min())

    def stats(self) -> dict:
        return {"experts": self.N, "eta": round(self.eta, 5),
                "alpha_fs": round(self.alpha_fs, 6), "rounds": self.rounds,
                "switches": self.switches, "switch_cost_total": round(self.cum_switch, 6),
                "regret_incl_switch": round(self.regret_so_far, 6),
                "weights_top": int(np.argmax(self.w)),
                "model_error_floor_2eps_mdl_M": round(2 * self.eps_mdl * self.rounds, 4)}
