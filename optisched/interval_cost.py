#!/usr/bin/env python3
"""Per-interval cost precomputation (proposal 5.1, 6.1, 7.1).

For every interval I=(i,j] with length w=j-i in [1, W_max] we precompute the
quantities that are *independent of congestion sigma*:

    residual_m(I,t) = A_m(I) - C_m(I,t)      per-owner residual miss count
    hot_m(I,t)      = |H(I,t) cap N_m|        per-owner hot-set size (for A')

where  A_m(I) = sum_{b in I} |N_{b,m}|                      (remote accesses to m)
       H(I,t) = top-n_hot remote nodes by template-t score  (the cache, A5)
       C_m(I,t) = sum_{u in H(I,t), owner(u)=m} freq_I(u)   (hot-set coverage)

This is exactly the §6.1 identity  sum_b miss_b(I) = A(I) - C(I)  carried per
owner and per allocation template.  Because residual/hot are sigma-independent,
the cost under *any* regime sigma (and the whole online expert library) is then a
cheap dot product -- see `cost_matrix`.

Selecting the top-n_hot set per growing interval is the only non-trivial offline
expense; the proposal bounds it at O(B*W_max*Fbar*log n_hot) and notes B is small
(1e2-1e3), so this runs in seconds offline.  We grow the window from each start i
and only pay the top-k when the window's unique remote set exceeds the cache
capacity (otherwise everything is cached and residual is zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .trace import Trace
from .calibration import CostModel


def weighted_template(model, sigma=None) -> List[np.ndarray]:
    """Theorem-B hot-set selection as a single template.

    The optimal in-window static cache selects the top-n_hot remote nodes by
    weighted frequency f_I(u)·c(u), where c(u) is the calibrated per-owner miss
    cost (congestion-scaled).  Encoding c as the per-owner selection weight makes
    interval_cost.precompute pick exactly that set (coverage/residual still use
    TRUE frequencies).  Reduces to top-frequency when c is constant (σ≡1,
    homogeneous owners).  Returns a 1-template list for `precompute(templates=...)`.
    """
    return [np.asarray(model.per_owner_miss_cost(sigma), dtype=np.float64)]


def make_templates(num_partitions: int, local_rank: int,
                   bias: float = 1.6) -> List[np.ndarray]:
    """Allocation templates (proposal 5.1, |T| = P).

    Template 0 is uniform (score = frequency).  The remaining templates each bias
    selection toward one remote owner (score = frequency * bias for that owner),
    pushing more cache capacity onto a congested owner -- the optimal analogue of
    GreenDyGNN's fixed 60%-bias cost-weighting.
    """
    templates = [np.ones(num_partitions, dtype=np.float64)]
    for p in range(num_partitions):
        if p == local_rank:
            continue
        w = np.ones(num_partitions, dtype=np.float64)
        w[p] = bias
        templates.append(w)
    return templates


@dataclass
class IntervalCosts:
    """Sigma-independent per-interval residual/hot tables.

    residual : float64[B, W_max, T, P]   residual misses per owner (inf if invalid)
    hot      : float64[B, W_max, T, P]   hot-set size per owner (0 if invalid)
    valid    : bool[B, W_max]            interval (i, i+w] fits in the epoch
    """

    residual: np.ndarray
    hot: np.ndarray
    valid: np.ndarray
    n_hot: int
    num_partitions: int
    templates: List[np.ndarray]

    @property
    def B(self) -> int:
        return self.residual.shape[0]

    @property
    def W_max(self) -> int:
        return self.residual.shape[1]

    @property
    def T(self) -> int:
        return self.residual.shape[2]

    # ----------------------------------------------------------------- costing
    def cost_matrix(self, model: CostModel, sigma: np.ndarray,
                    variant: str = "A"):
        """Cost of every interval under congestion sigma (vectorised).

        Returns (cost[B, W_max], best_template[B, W_max]); invalid intervals are
        +inf.  `variant` selects the rebuild term:
          'A'  -- length-only calibrated rebuild  alpha*(a+b*w**c)   (MAIN)
          'Ap' -- full-refresh certified upper bound (proposal 5.2): the rebuild
                  re-sends the whole hot set, charged per owner.
        cost is in *time* units (seconds); energy = model.energy(cost).
        """
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if sigma.size != self.num_partitions:
            raise ValueError("sigma length must equal num_partitions")
        w_weight = (sigma * model.t_miss).reshape(1, 1, 1, -1)  # per-owner miss weight
        miss = np.einsum("bwtp,xyzp->bwt", self.residual, w_weight)  # [B,W,T]

        lengths = np.arange(1, self.W_max + 1, dtype=np.float64)
        if variant == "A":
            rebuild = model.rebuild_term(lengths).reshape(1, -1, 1)  # [1,W,1]
            total = miss + rebuild
        elif variant == "Ap":
            # Full-refresh: lambda_init(m)*1[hot_m>0] + nu_m*hot_m, on critical path.
            # Reuse t_miss as the per-node transfer slope and a small per-owner
            # initiation = t_miss (one extra RPC) -- self-contained upper bound.
            init = model.t_miss.reshape(1, 1, 1, -1)
            slope = model.t_miss.reshape(1, 1, 1, -1)
            fr = np.einsum("bwtp,xyzp->bwt", (self.hot > 0).astype(np.float64), init) \
                + np.einsum("bwtp,xyzp->bwt", self.hot, slope)
            total = miss + model.alpha * fr
        else:
            raise ValueError(f"unknown variant {variant!r}")

        # mask invalid intervals
        total = np.where(self.valid[:, :, None], total, np.inf)
        best_t = np.argmin(total, axis=2)
        cost = np.take_along_axis(total, best_t[:, :, None], axis=2)[:, :, 0]
        return cost, best_t


def precompute(trace: Trace, model: CostModel, n_hot: int,
               w_max: Optional[int] = None,
               templates: Optional[Sequence[np.ndarray]] = None,
               start_stride: int = 1) -> IntervalCosts:
    """Build the sigma-independent interval tables for a trace.

    Parameters
    ----------
    n_hot       : per-worker cache capacity in nodes (A5/A6).
    w_max       : max window length; defaults to model.w_max.
    templates   : allocation templates; defaults to uniform only (boundaries-only).
                  Pass make_templates(...) to enable joint boundary+allocation.
    start_stride: subsample window starts for very large traces (gate only); the
                  DP then operates on the strided lattice.  Default 1 (exact).
    """
    B = trace.num_batches
    W_max = int(w_max or model.w_max)
    P = trace.num_partitions
    if templates is None:
        templates = [np.ones(P, dtype=np.float64)]
    templates = [np.asarray(t, dtype=np.float64) for t in templates]
    T = len(templates)
    tmpl = np.stack(templates)  # [T, P]

    residual = np.full((B, W_max, T, P), np.inf, dtype=np.float64)
    hot = np.zeros((B, W_max, T, P), dtype=np.float64)
    valid = np.zeros((B, W_max), dtype=bool)

    for i in range(0, B, start_stride):
        cnt = {}                      # node -> within-window batch frequency
        own = {}                      # node -> owner partition
        Am = np.zeros(P, dtype=np.float64)     # remote accesses per owner
        distm = np.zeros(P, dtype=np.float64)  # distinct nodes per owner
        wmax_here = min(W_max, B - i)
        for l in range(wmax_here):
            b = i + l
            nb, ob = trace.batch(b)
            nb_l = nb.tolist()
            ob_l = ob.tolist()
            for node, owner in zip(nb_l, ob_l):
                c = cnt.get(node)
                if c is None:
                    cnt[node] = 1
                    own[node] = owner
                    distm[owner] += 1.0
                else:
                    cnt[node] = c + 1
            if ob.size:
                np.add.at(Am, ob, 1.0)

            U = len(cnt)
            valid[i, l] = True
            if U <= n_hot:
                # Entire unique set fits in cache: zero residual, hot = distinct.
                residual[i, l, :, :] = 0.0
                hot[i, l, :, :] = distm[None, :]
                continue

            # Top-n_hot selection per template.
            nodes_arr = np.fromiter(cnt.keys(), dtype=np.int64, count=U)
            counts_arr = np.fromiter(cnt.values(), dtype=np.float64, count=U)
            owners_arr = np.fromiter((own[n] for n in nodes_arr), dtype=np.int64,
                                     count=U)
            for t in range(T):
                score = counts_arr * tmpl[t, owners_arr]
                # indices of the n_hot highest scores (unordered top-k)
                top = np.argpartition(score, U - n_hot)[U - n_hot:]
                sel_owners = owners_arr[top]
                sel_counts = counts_arr[top]
                cover = np.bincount(sel_owners, weights=sel_counts, minlength=P)
                hcount = np.bincount(sel_owners, minlength=P).astype(np.float64)
                residual[i, l, t, :] = Am - cover
                hot[i, l, t, :] = hcount

    # Numerical guard: residual misses are non-negative on valid intervals.
    finite = np.isfinite(residual)
    residual[finite] = np.clip(residual[finite], 0.0, None)
    return IntervalCosts(residual, hot, valid, n_hot, P, list(templates))
