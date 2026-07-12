#!/usr/bin/env python3
"""Schedule solvers (proposal 6.1, 6.2, 6.3, 7.2).

Given a per-interval cost matrix `cost[i][w-1]` = cost of the window covering
batches (i, i+w] (left boundary i in 0..B-1, length w in 1..W_max), this module
provides:

  solve_optionA       Theorem-1 exact non-uniform DP (O(B*W_max)) + backpointers.
  oracle_uniform      best uniform-W schedule under the SAME cost (Theorem-2 base).
  brute_force         exhaustive optimum for tiny B (validation only).
  per_position_w      per-position locally-optimal window length (heterogeneity).
  solve_optionB       Theorem-3 delta-exact interval-transition DP (refinement).

The cost matrix is produced by `interval_cost.IntervalCosts.cost_matrix`.
Invalid intervals carry +inf, so the solvers never select them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Schedule:
    """A solved schedule over one epoch's B batches.

    boundaries : list of cut points 0 = j_0 < j_1 < ... < j_K = B
    windows    : list of (start, length) per window, in order
    templates  : chosen allocation template index per window (Option A)
    cost       : total schedule cost (time units; energy = model.energy(cost))
    """

    boundaries: List[int]
    windows: List[Tuple[int, int]]
    templates: List[int]
    cost: float

    def window_at(self, batch_index: int) -> Tuple[int, int, int]:
        """Return (start, length, template) of the window containing batch_index."""
        for (s, w), t in zip(self.windows, self.templates):
            if s <= batch_index < s + w:
                return s, w, t
        s, w = self.windows[-1]
        return s, w, self.templates[-1]


def solve_optionA(cost: np.ndarray, best_t: Optional[np.ndarray] = None,
                  w_max: Optional[int] = None) -> Schedule:
    """Theorem-1 exact DP over contiguous partitions (proposal 7.2).

    OPT[0]=0;  OPT[j] = min_{j-W_max <= i < j} OPT[i] + cost[i][j-i-1].
    """
    B, W = cost.shape
    W_max = int(w_max or W)
    INF = np.inf
    opt = np.full(B + 1, INF)
    back = np.full(B + 1, -1, dtype=np.int64)
    opt[0] = 0.0
    for j in range(1, B + 1):
        lo = max(0, j - W_max)
        # candidate predecessors i in [lo, j-1]; window length w = j-i in [1, j-lo]
        best = INF
        bi = -1
        for i in range(lo, j):
            w = j - i
            c = cost[i, w - 1]
            if not np.isfinite(c):
                continue
            v = opt[i] + c
            if v < best:
                best = v
                bi = i
        opt[j] = best
        back[j] = bi
    if not np.isfinite(opt[B]):
        raise RuntimeError("no feasible schedule (check W_max vs B and cost validity)")

    # recover boundaries
    bounds = [B]
    j = B
    while j > 0:
        i = int(back[j])
        bounds.append(i)
        j = i
    bounds.reverse()
    windows, templates = [], []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        windows.append((s, e - s))
        templates.append(int(best_t[s, e - s - 1]) if best_t is not None else 0)
    return Schedule(bounds, windows, templates, float(opt[B]))


def oracle_uniform(cost: np.ndarray, w_grid: List[int],
                   best_t: Optional[np.ndarray] = None
                   ) -> Tuple[Schedule, Dict[int, float]]:
    """Best uniform-W schedule by exhaustive replay (proposal 8 baseline).

    For each W in w_grid the epoch is partitioned into windows of length W (last
    possibly shorter); returns the cheapest such schedule and the per-W totals.
    Evaluated under the SAME cost matrix as the DP -> apples-to-apples (§8).
    """
    B, Wm = cost.shape
    per_w: Dict[int, float] = {}
    best_schedule = None
    for W in w_grid:
        if W < 1 or W > Wm:
            continue
        total = 0.0
        windows, templates, bounds = [], [], [0]
        i = 0
        feasible = True
        while i < B:
            w = min(W, B - i)
            c = cost[i, w - 1]
            if not np.isfinite(c):
                feasible = False
                break
            total += c
            windows.append((i, w))
            templates.append(int(best_t[i, w - 1]) if best_t is not None else 0)
            i += w
            bounds.append(i)
        if not feasible:
            continue
        per_w[W] = float(total)
        if best_schedule is None or total < best_schedule.cost:
            best_schedule = Schedule(bounds, windows, templates, float(total))
    if best_schedule is None:
        raise RuntimeError("no feasible uniform schedule over the given grid")
    return best_schedule, per_w


def per_position_w(cost: np.ndarray, w_max: Optional[int] = None) -> np.ndarray:
    """Per-position locally-optimal window length W*(i) (heterogeneity metric).

    For each start position i, the length minimising the per-batch *amortised*
    cost  cost[i][w-1]/w.  Its across-position (and across-epoch) variance is the
    temporal-heterogeneity-of-locality metric in proposal §8.
    """
    B, W = cost.shape
    W_max = int(w_max or W)
    out = np.ones(B, dtype=np.int64)
    for i in range(B):
        best, bw = np.inf, 1
        for w in range(1, min(W_max, B - i) + 1):
            c = cost[i, w - 1]
            if not np.isfinite(c):
                break
            amort = c / w
            if amort < best:
                best, bw = amort, w
        out[i] = bw
    return out


def brute_force(cost: np.ndarray, w_max: Optional[int] = None) -> float:
    """Exhaustive optimum over all contiguous partitions (validation, tiny B)."""
    B, W = cost.shape
    W_max = int(w_max or W)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best(i: int) -> float:
        if i == B:
            return 0.0
        v = np.inf
        for w in range(1, min(W_max, B - i) + 1):
            c = cost[i, w - 1]
            if np.isfinite(c):
                v = min(v, c + best(i + w))
        return v

    return float(best(0))


# --------------------------------------------------------------------- Option B
def solve_optionB(trace, model, n_hot: int, sigma: np.ndarray,
                  w_max: Optional[int] = None) -> Schedule:
    """Theorem-3 delta-exact interval-transition DP (proposal 6.3).

    The rebuild for window (i,j] charges only the nodes newly entering the hot
    set relative to the previous window (k,i]:  Delta = H((i,j]) \\ H((k,i]).
    State is the consecutive boundary pair; complexity O(B*W_max^2).  Uses the
    uniform template (allocation is an Option-A feature).  Intended for the
    H-slack measurement on small/medium traces, not the large-scale gate.
    """
    from .interval_cost import precompute

    B = trace.num_batches
    W_max = int(w_max or model.w_max)
    P = trace.num_partitions
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)

    # Residual miss time per interval (uniform template) -- same as Option A.
    ic = precompute(trace, model, n_hot, w_max=W_max)
    miss_cost, _ = _miss_only(ic, model, sigma)

    # Hot-set node ids per interval (uniform template), built by growing windows.
    hotset: Dict[Tuple[int, int], np.ndarray] = {}
    owner_of: Dict[int, int] = {}
    for i in range(B):
        cnt: Dict[int, float] = {}
        for l in range(min(W_max, B - i)):
            nb, ob = trace.batch(i + l)
            for node, owner in zip(nb.tolist(), ob.tolist()):
                cnt[node] = cnt.get(node, 0.0) + 1.0
                owner_of[node] = owner
            U = len(cnt)
            nodes_arr = np.fromiter(cnt.keys(), dtype=np.int64, count=U)
            if U <= n_hot:
                hotset[(i, l + 1)] = nodes_arr
            else:
                counts_arr = np.fromiter(cnt.values(), dtype=np.float64, count=U)
                top = np.argpartition(counts_arr, U - n_hot)[U - n_hot:]
                hotset[(i, l + 1)] = nodes_arr[top]

    def delta_rebuild(prev_hot: Optional[np.ndarray], cur_hot: np.ndarray) -> float:
        if prev_hot is None or prev_hot.size == 0:
            new = cur_hot
        else:
            new = cur_hot[~np.isin(cur_hot, prev_hot)]
        if new.size == 0:
            return 0.0
        ow = np.fromiter((owner_of[n] for n in new), dtype=np.int64, count=new.size)
        cnt_owner = np.bincount(ow, minlength=P).astype(np.float64)
        # init (one RPC per touched owner) + per-node slope, weighted by sigma.
        touched = (cnt_owner > 0).astype(np.float64)
        t = (sigma * model.t_miss)
        return model.alpha * float((touched * t).sum() + (cnt_owner * t).sum())

    # DP over (prev_boundary k, cur_boundary i) -> extend to j.
    # opt[(k,i)] = min cost of a schedule covering [0,i] whose last window is (k,i].
    INF = np.inf
    opt: Dict[Tuple[int, int], float] = {}
    back: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    # seed: first window (0, i] has no previous hot set
    for w in range(1, min(W_max, B) + 1):
        key = (0, w)
        opt[key] = delta_rebuild(None, hotset[(0, w)]) + miss_cost[0, w - 1]
        back[key] = None

    order = sorted(opt.keys(), key=lambda kv: kv[1])
    # process boundaries in increasing end position
    for end in range(1, B + 1):
        states = [(k, i) for (k, i) in opt.keys() if i == end]
        for (k, i) in states:
            base = opt[(k, i)]
            if not np.isfinite(base):
                continue
            prev_hot = hotset[(k, i - k)]   # last window (k, i] -> start k, length i-k
            for w in range(1, min(W_max, B - i) + 1):
                j = i + w
                d = delta_rebuild(prev_hot, hotset[(i, w)])
                v = base + d + miss_cost[i, w - 1]
                key = (i, j)
                if key not in opt or v < opt[key]:
                    opt[key] = v
                    back[key] = (k, i)

    # best terminal state ending at B
    terminals = [(k, i) for (k, i) in opt.keys() if i == B]
    if not terminals:
        raise RuntimeError("Option-B DP found no feasible schedule")
    best_key = min(terminals, key=lambda key: opt[key])
    total = opt[best_key]

    # recover
    bounds = [B]
    key = best_key
    while key is not None:
        k, i = key
        bounds.append(k)
        key = back[key]
    bounds = sorted(set(bounds))
    windows = [(bounds[t], bounds[t + 1] - bounds[t]) for t in range(len(bounds) - 1)]
    templates = [0] * len(windows)
    return Schedule(bounds, windows, templates, float(total))


@dataclass
class DecoupledResult:
    """Owner-decoupled schedule (Theorem G): one independent schedule per owner."""
    total_cost: float
    per_owner: Dict[int, Schedule]
    budgets: Dict[int, int]


def solve_owner_decoupled(trace, model, budgets: Dict[int, int], sigma=None,
                          w_max: Optional[int] = None) -> "DecoupledResult":
    """Theorem G: exact owner-decoupled DP for FIXED per-owner budgets.

    Each owner m schedules its own rebuild boundaries over its own sub-trace with
    its own budget n_hot,m, independently of other owners (its cost depends only
    on its trace, boundaries, budget, coefficients).  Total = Σ_m OPT_m.  This
    removes the owner-synchronization slack of a single global boundary sequence.
    """
    from .interval_cost import precompute
    P = trace.num_partitions
    W_max = int(w_max or model.w_max)
    sigma = np.ones(P) if sigma is None else np.asarray(sigma, dtype=np.float64)
    per_owner: Dict[int, Schedule] = {}
    total = 0.0
    for m, nh in budgets.items():
        sub = trace.restrict_owner(m)
        ic = precompute(sub, model, int(nh), w_max=W_max)
        cost, bt = ic.cost_matrix(model, sigma, variant="A")
        sched = solve_optionA(cost, bt, w_max=W_max)
        per_owner[m] = sched
        total += sched.cost
    return DecoupledResult(total_cost=float(total), per_owner=per_owner,
                           budgets=dict(budgets))


def solve_budget_templates(trace, model, templates: List[Dict[int, int]],
                           sigma=None, w_max: Optional[int] = None):
    """Object 2 (Architecture §14): exact owner-decoupled optimum over a FINITE
    static budget-template set.  Returns (best DecoupledResult, all results).
    Exact only within the template set -- not the unrestricted budget optimum."""
    results = [solve_owner_decoupled(trace, model, t, sigma, w_max) for t in templates]
    best = min(results, key=lambda r: r.total_cost)
    return best, results


def _miss_only(ic, model, sigma):
    """Residual-miss time per interval (no rebuild term), uniform template."""
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    w_weight = (sigma * model.t_miss).reshape(1, 1, 1, -1)
    miss = np.einsum("bwtp,xyzp->bwt", ic.residual, w_weight)
    miss = np.where(ic.valid[:, :, None], miss, np.inf)
    best_t = np.argmin(miss, axis=2)
    cost = np.take_along_axis(miss, best_t[:, :, None], axis=2)[:, :, 0]
    return cost, best_t
