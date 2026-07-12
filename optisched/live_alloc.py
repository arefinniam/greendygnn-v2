"""Live floor-guided owner-aware cache allocation (Thm G, payload-dominated form).

The offline ablation (`allocation_ablation.py`) computes per-owner cache budgets
n_m by marginal-greedy on the calibrated energy E = sum_m[eps_init*R_m + kappa_m*d*Q_m].
In the payload-dominated regime the campaign measured (eta >> 1, n* = 13-80 rows ≪
fetches, so the eps_init*R initiation term is negligible), the refetch energy of
owner m is kappa_m*d*Q_m where Q_m = remote accesses to owner-m nodes NOT cached.
Caching owner m's top-n_m most-accessed nodes removes sum(their access counts) from
Q_m, so the marginal energy reduction of the (j+1)-th cache block given to owner m is

        dE_m(j) = kappa_m * d * count_m[j]        (count_m sorted descending)

which is monotone decreasing in j. Block-1 marginal-greedy therefore always caches
the not-yet-cached node u with the largest kappa_{owner(u)} * count_u, i.e. it is
EXACTLY the top-n_hot selection by score(u) = kappa_{owner(u)} * count_u. The
resulting per-owner budget n_m is the number of selected owner-m nodes; the
footprint cap n_m <= |U_m| holds automatically (an unaccessed node is never scored).

This module is the single source of truth for that selection so it can be unit
tested against the offline marginal-greedy. It is default-off: with kappa=None the
caller keeps the legacy uniform top-frequency behaviour.
"""
from typing import Dict, Optional, Tuple
import numpy as np


def kappa_vector(kappa: Dict[int, float], num_owners: int, ref: str = "min") -> np.ndarray:
    """Dense per-owner weight vector, normalised so the cheapest owner = 1.0.

    Normalisation is scale-free: scaling all kappa multiplies every score equally
    and leaves the top-k selection unchanged, so only relative owner cost matters
    (which is exactly what the floor's rho is invariant to under A4)."""
    w = np.ones(num_owners, dtype=np.float64)
    for m, k in kappa.items():
        if 0 <= m < num_owners:
            w[m] = float(k)
    pos = w[w > 0]
    if pos.size:
        denom = pos.min() if ref == "min" else 1.0
        if denom > 0:
            w = w / denom
    return w


def select_owner_budgeted(
    unique_nodes: np.ndarray,
    counts: np.ndarray,
    owners: np.ndarray,
    n_hot: int,
    kappa: Optional[Dict[int, float]],
    num_owners: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Floor-guided owner-aware hot-node selection.

    Returns (selected_node_ids, budgets) where budgets[m] = n_m, the realised
    per-owner cache budget (Thm G split). With kappa=None this reduces to plain
    top-n_hot by access count (uniform allocation), so it is a faithful superset
    of the legacy path.
    """
    n = unique_nodes.shape[0]
    k = min(int(n_hot), n)
    if k <= 0:
        return unique_nodes[:0], {}
    if kappa is None:
        scores = counts.astype(np.float64)
    else:
        w = kappa_vector(kappa, num_owners)
        scores = counts.astype(np.float64) * w[owners]
    # top-k by score (argpartition for O(n), then materialise the indices)
    if k < n:
        top = np.argpartition(scores, n - k)[n - k:]
    else:
        top = np.arange(n)
    sel_owners = owners[top]
    budgets = {int(m): int(c) for m, c in zip(*np.unique(sel_owners, return_counts=True))}
    return unique_nodes[top], budgets
