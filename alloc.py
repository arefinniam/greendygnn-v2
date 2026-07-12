#!/usr/bin/env python3
"""Analytic per-owner cache allocation (V2_SPEC I3).

Single entry point ``select_owner_budgets``: top-n_hot hot-node selection scored
by ``khat[owner(u)] * count(u)`` — block-1 marginal-greedy on the payload-dominated
refetch energy, which is EXACTLY optimal in that regime (see
optisched/live_alloc.py, whose ``select_owner_budgeted`` implements the selection;
this module wraps it — it does NOT fork the logic).

Why the cap (khat_cap, default 8.0)
-----------------------------------
The 2026-06-29 single-point heterogeneity validation showed that feeding a raw,
very large owner-cost ratio (kappa=16 vs a count distribution topping out ~15,
later kappa=193) makes the marginal-greedy cache the expensive owner EXCLUSIVELY,
abandoning frequently-reused nodes of the fast owners: hit rate collapsed
(24.5%->18.3%) and measured energy got WORSE than uniform. Allocation must react
to *relative* owner cost without ever fully starving fast owners on a noisy or
transient estimate, so the effective khat is clamped to [1, khat_cap]. The
calibrated sweep that validated the lever (gain 21.5%->43.7% for measured kappa
9.6->191) still saturates near the cap because once the slow owner is a few times
more expensive than the rest, the selection ordering barely changes with larger
kappa.

khat=None preserves the legacy pure top-count (uniform) behaviour bit-for-bit.
"""
from typing import Dict, Optional, Tuple

import numpy as np

from optisched.live_alloc import select_owner_budgeted

DEFAULT_KHAT_CAP = 8.0


def cap_khat(khat: Optional[Dict[int, float]],
             khat_cap: float = DEFAULT_KHAT_CAP) -> Optional[Dict[int, float]]:
    """Clamp per-owner cost estimates to [1.0, khat_cap].

    khat is normalised so the fastest owner is 1.0 (controller I2 contract);
    values below 1 are estimation noise and are floored, values above the cap
    are clamped (see module docstring for the measured failure mode).
    """
    if khat is None:
        return None
    return {int(pid): float(min(max(1.0, float(k)), khat_cap))
            for pid, k in khat.items()}


def select_owner_budgets(
    unique_nodes: np.ndarray,
    counts: np.ndarray,
    owners: np.ndarray,
    n_hot: int,
    khat: Optional[Dict[int, float]],
    num_owners: Optional[int] = None,
    khat_cap: float = DEFAULT_KHAT_CAP,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Owner-aware hot-set selection (I3).

    Parameters
    ----------
    unique_nodes, counts, owners : parallel arrays describing the candidate hot
        set of the upcoming window (global node id, access count, owner pid).
    n_hot  : cache capacity in rows.
    khat   : per-owner relative cost (fastest==1.0) or None for uniform top-count.
    num_owners : P (partition count); owner pids are 0..P-1. Optional — inferred
        from the owners array and khat keys when omitted (the prefetcher calls
        with 5 positional args).
    khat_cap : safety clamp, see module docstring.

    Returns
    -------
    (selected_node_ids, budgets) with budgets[pid] = number of selected rows
    owned by pid (the realised Thm-G split; logged by the builder).
    """
    if num_owners is None:
        hi = int(np.max(owners)) if len(owners) else 0
        if khat:
            hi = max(hi, max(int(k) for k in khat))
        num_owners = hi + 1
    return select_owner_budgeted(
        unique_nodes, counts, owners, n_hot, cap_khat(khat, khat_cap), num_owners)
