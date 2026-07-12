#!/usr/bin/env python3
r"""Feasible-transfer communication floor and near-floor decomposition.

Implements the new spine of the OptiSched-GNN Architecture:

  Theorem A (exact floor):  E_c(Π) >= L_0  for every feasible policy whose bulk
    transfer from owner m carries <= C_max[m] rows, where
        L_0 = Σ_m ε_init(m)·⌈|U_m|/C_max[m]⌉  +  Σ_m κ_m·d·|U_m|
              \_____ initiation floor _____/     \____ payload floor ____/
    (each distinct remote row delivered >=1 time; the transfer cap forces a
    minimum number of RPCs).  L_0 depends only on the trace footprints U_m, the
    caps, and the measured coefficients -- no schedule needed.

  Theorem F (near-floor decomposition):  for a realised windowed schedule with
    R_win bulk transfers and Q_win rows transferred,
        γ_R = R_win/R_0   (initiation inflation),   φ = Q_win/Q_0   (payload inflation)
        η   = (Σ κ_m d|U_m|)/(Σ ε_init(m)⌈|U_m|/C_max⌉)   (payload:init at floor)
        ρ   = E_win/L_0 = (γ_R + η·φ)/(1+η)               (distance to floor)
    In the initiation-dominated regime (η ≪ 1):  ρ = γ_R + O(η·φ).
    Crossover n* = 1/θ = ε_init/(κ d); n̄ = Q_0/R_0 (avg payload per floor transfer).

  Lemma D (lifetime-fit payload attainment): if every node's active lifetime is
    contiguous and the simultaneously-live set never exceeds capacity, a delta
    policy fetches each node once ⇒ Q_win = |U|, φ = 1.

The floor is CONDITIONAL on the fixed trace/partition/feature-rep/transport and
the MEASURED caps + coefficients -- it is the floor for the measured instance, not
a universal bound (Architecture §4 interpretation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .trace import Trace
from .calibration import CostModel


# ----------------------------------------------------------------- footprints
def footprints(trace: Trace) -> dict:
    """Per-owner distinct remote footprint |U_m|, total |U|, total accesses A."""
    P = trace.num_partitions
    Um = [set() for _ in range(P)]
    A = 0
    for b in range(trace.num_batches):
        nodes, owners = trace.batch(b)
        A += int(nodes.size)
        for u, m in zip(nodes.tolist(), owners.tolist()):
            Um[m].add(u)
    sizes = np.array([len(s) for s in Um], dtype=np.float64)
    return {"U_m": sizes, "U": float(sizes.sum()), "A": float(A), "sets": Um}


# ----------------------------------------------------------------- Theorem A
@dataclass
class Floor:
    L0: float                 # communication-energy floor (Joules)
    R0: float                 # initiation floor: Σ ⌈|U_m|/C_max⌉ (transfers)
    Q0: float                 # payload floor: |U| (rows)
    init_energy: float        # Σ ε_init(m)·⌈|U_m|/C_max⌉
    payload_energy: float     # Σ κ_m·d·|U_m|
    U_m: np.ndarray
    eta: float                # payload:initiation ratio at the floor
    theta_mean: float         # κ·d/ε_init (mean over owners)
    n_star_mean: float        # crossover 1/θ (rows)
    nbar0: float              # Q0/R0  (avg payload per floor transfer)


def communication_floor(trace: Trace, model: CostModel) -> Floor:
    """Theorem A: the exact feasible-transfer communication floor (Joules)."""
    fp = footprints(trace)
    Um = fp["U_m"]
    cap = np.maximum(model.c_max, 1.0)
    rpcs = np.ceil(Um / cap)                       # ⌈|U_m|/C_max⌉ per owner
    init_e = float((model.eps_init * rpcs).sum())
    pay_e = float((model.kappa * model.d * Um).sum())
    L0 = init_e + pay_e
    R0 = float(rpcs.sum())
    Q0 = float(Um.sum())
    eta = pay_e / init_e if init_e > 0 else np.inf
    theta = float(np.mean(model.kappa * model.d / np.maximum(model.eps_init, 1e-300)))
    return Floor(L0=L0, R0=R0, Q0=Q0, init_energy=init_e, payload_energy=pay_e,
                 U_m=Um, eta=eta, theta_mean=theta,
                 n_star_mean=(1.0 / theta if theta > 0 else np.inf),
                 nbar0=(Q0 / R0 if R0 > 0 else np.inf))


# --------------------------------------------- per-window hot set (Theorem B)
def _window_hot_set(trace: Trace, start: int, length: int, n_hot: int,
                    owner_cost: np.ndarray):
    """Top-n_hot remote nodes in (start, start+length] by weighted frequency
    f_I(u)·c(owner(u)) (Theorem B).  Returns {node: owner} for the selected set."""
    cnt, own = {}, {}
    for b in range(start, min(start + length, trace.num_batches)):
        nodes, owners = trace.batch(b)
        for u, m in zip(nodes.tolist(), owners.tolist()):
            cnt[u] = cnt.get(u, 0) + 1
            own[u] = m
    U = len(cnt)
    if U == 0:
        return {}
    if U <= n_hot:
        return own
    nodes_arr = np.fromiter(cnt.keys(), dtype=np.int64, count=U)
    counts = np.fromiter(cnt.values(), dtype=np.float64, count=U)
    owners = np.fromiter((own[u] for u in nodes_arr), dtype=np.int64, count=U)
    score = counts * owner_cost[owners]
    top = np.argpartition(score, U - n_hot)[U - n_hot:]
    return {int(nodes_arr[i]): int(owners[i]) for i in top}


# ------------------------------------- schedule transfer accounting (Thm F)
def schedule_transfers(trace: Trace, windows: Sequence[Tuple[int, int]],
                       model: CostModel, n_hot: int, sigma=None,
                       rebuild: str = "delta") -> dict:
    """Realised communication of running a windowed schedule.

    For each window: rebuild the (weighted) hot set, then serve residual misses
    on demand per (batch, owner).  `rebuild`='full' re-sends the whole hot set;
    'delta' sends only nodes new vs the previous window's hot set.  Returns
    per-owner R (transfers) and Q (rows), the totals, and E_win (Joules).
    """
    P = trace.num_partitions
    cap = np.maximum(model.c_max, 1.0)
    owner_cost = model.per_owner_miss_cost(sigma)
    R = np.zeros(P)
    Q = np.zeros(P)
    prev_hot: Dict[int, int] = {}
    for (s, w) in windows:
        hot = _window_hot_set(trace, s, w, n_hot, owner_cost)
        # rebuild transfers
        if rebuild == "delta":
            fetch = {u: m for u, m in hot.items() if u not in prev_hot}
        else:
            fetch = hot
        rows_m = np.zeros(P)
        for u, m in fetch.items():
            rows_m[m] += 1
        for m in range(P):
            if rows_m[m] > 0:
                Q[m] += rows_m[m]
                R[m] += np.ceil(rows_m[m] / cap[m])
        prev_hot = hot
        # on-demand residual misses within the window (served per batch/owner)
        hot_set = set(hot.keys())
        for b in range(s, min(s + w, trace.num_batches)):
            nodes, owners = trace.batch(b)
            miss_m = np.zeros(P)
            for u, m in zip(nodes.tolist(), owners.tolist()):
                if u not in hot_set:
                    miss_m[m] += 1
            for m in range(P):
                if miss_m[m] > 0:
                    Q[m] += miss_m[m]
                    R[m] += np.ceil(miss_m[m] / cap[m])
    E_win = float((model.eps_init * R).sum() + (model.kappa * model.d * Q).sum())
    return {"R_m": R, "Q_m": Q, "R_win": float(R.sum()), "Q_win": float(Q.sum()),
            "E_win": E_win}


# --------------------------------------------------- Theorem F decomposition
@dataclass
class NearFloor:
    rho: float            # distance to floor E_win/L_0
    rho_decomp: float     # (γ_R + η φ)/(1+η)  -- should match rho (homogeneous)
    gamma_R: float        # initiation inflation R_win/R_0
    phi: float            # payload inflation Q_win/Q_0
    eta: float            # payload:initiation at floor
    nbar: float           # realised payload per transfer Q_win/R_win
    n_star: float         # crossover 1/θ
    initiation_dominated: bool


def near_floor(floor: Floor, transfers: dict, model: CostModel) -> NearFloor:
    """Theorem F: decompose the realised schedule's distance to the floor."""
    R_win, Q_win, E_win = transfers["R_win"], transfers["Q_win"], transfers["E_win"]
    gamma_R = R_win / floor.R0 if floor.R0 > 0 else np.inf
    phi = Q_win / floor.Q0 if floor.Q0 > 0 else np.inf
    eta = floor.eta
    rho = E_win / floor.L0 if floor.L0 > 0 else np.inf
    rho_decomp = (gamma_R + eta * phi) / (1.0 + eta) if np.isfinite(eta) else phi
    nbar = Q_win / R_win if R_win > 0 else np.inf
    return NearFloor(rho=rho, rho_decomp=rho_decomp, gamma_R=gamma_R, phi=phi,
                     eta=eta, nbar=nbar, n_star=floor.n_star_mean,
                     initiation_dominated=bool(eta < 1.0))


# --------------------------------------------------- Lemma D: lifetime-fit
def lifetime_fit(trace: Trace, windows: Sequence[Tuple[int, int]],
                 n_hot: int) -> dict:
    """Check the sufficient condition for φ=1 (Lemma D): every node's per-window
    lifetime is contiguous AND the simultaneously-live set never exceeds n_hot.

    Returns {contiguous: bool, max_live: int, capacity_ok: bool, attainable: bool}.
    """
    # which windows each node appears in
    appears: Dict[int, List[int]] = {}
    for k, (s, w) in enumerate(windows):
        seen = set()
        for b in range(s, min(s + w, trace.num_batches)):
            nodes, _ = trace.batch(b)
            seen.update(nodes.tolist())
        for u in seen:
            appears.setdefault(u, []).append(k)
    contiguous = True
    for u, ks in appears.items():
        if ks != list(range(ks[0], ks[-1] + 1)):
            contiguous = False
            break
    # max simultaneously-live nodes (contiguous-lifetime interpretation)
    K = len(windows)
    live = np.zeros(K + 1, dtype=np.int64)
    for u, ks in appears.items():
        live[ks[0]] += 1
        live[ks[-1] + 1] -= 1
    max_live = int(np.cumsum(live)[:K].max()) if K else 0
    capacity_ok = max_live <= n_hot
    return {"contiguous": bool(contiguous), "max_live": max_live,
            "capacity_ok": bool(capacity_ok),
            "attainable": bool(contiguous and capacity_ok)}


# --------------------------------------------------- working-set drift β (§15)
def working_set_drift(trace: Trace, w: int) -> float:
    """β(w) = Σ_k |U_k(w)| / |U|  for fixed window length w.  β≈1 ⇒ nodes appear
    in temporally compact blocks; large β ⇒ heavy reuse/churn across windows.
    A structural predictor of φ, γ_R, and the non-uniform scheduling gain."""
    fp = footprints(trace)
    total_U = fp["U"]
    if total_U == 0:
        return 1.0
    M = trace.num_batches
    s = 0
    acc = 0
    while s < M:
        seg = set()
        for b in range(s, min(s + w, M)):
            nodes, _ = trace.batch(b)
            seg.update(nodes.tolist())
        acc += len(seg)
        s += w
    return float(acc / total_U)
