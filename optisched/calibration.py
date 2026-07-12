#!/usr/bin/env python3
"""Calibrated, length-parameterized cost model (proposal A3, eq. in 5.1).

The cost of any window/interval I=(i,j] of length w under congestion sigma is

    T_A(I; sigma) = w*T_base                      (schedule-invariant; reporting)
                  + alpha * T_rebuild(w)          (one rebuild, length-only)
                  + sum_m sigma_m * t_miss_m * residual_m(I)   (trace-exact misses)

with  T_rebuild(w) = a + b * w**c   (0 < c < 1, sublinear; GreenDyGNN Alg.1 Ph.2),
and   residual_m(I) = A_m(I) - C_m(I)  the per-owner residual miss count, which is
trace-exact and supplied by interval_cost (so the logistic hit-rate fit h(W) is
removed from the quantity we optimise — proposal 5.1).

Energy is reported as E = P_bar * T  (A4 energy-time proportionality).

This module only holds parameters and the two scalar functions T_rebuild / energy;
all the heavy trace arithmetic lives in interval_cost.  Parameters can be:
  * the documented defaults (derived from the three papers), or
  * fitted from a one-time profiling run via `CostModel.fit_*`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np


# Window length is bounded by device memory; GreenDyGNN caps the action set at 128.
DEFAULT_W_MAX = 128


@dataclass
class CostModel:
    """Calibrated cost-model parameters.

    Indexing convention: every per-owner array has length ``num_partitions`` and
    is indexed by *global partition id*.  The local partition's entry is unused
    (no node is remote to its own owner); it is kept zero so owner ids line up
    with the partition book everywhere else in the codebase.

    Attributes
    ----------
    a, b, c     : rebuild model  T_rebuild(w) = a + b*w**c   (seconds)
    alpha       : fraction of rebuild time on the critical path (async pipeline)
    t_miss      : per-owner mean per-node miss-RPC latency (seconds), length P
    num_partitions : P
    p_bar       : mean active power (Watts) -- energy = p_bar * time (A4)
    t_base      : irreducible per-step compute + AllReduce time (seconds)
    eps_mdl     : additive cost-estimate error bound (the calibrated ~2.8% fit);
                  used by the online regret accounting (Theorem 5).
    w_max       : maximum window length (A6).
    """

    a: float = 2.0e-3
    b: float = 1.2e-3
    c: float = 0.6
    alpha: float = 0.35
    t_miss: np.ndarray = field(default_factory=lambda: np.array([]))
    num_partitions: int = 4
    p_bar: float = 190.0
    t_base: float = 8.0e-3
    eps_mdl: float = 0.028
    w_max: int = DEFAULT_W_MAX

    # --- Structural communication-energy coefficients (Architecture §2-3) ---
    # E_c(Π) = Σ_m ε_init(m)·R_m + Σ_m κ_m·d·Q_m  (initiation + payload energy);
    # the feasible-transfer floor L_0 (Theorem A) is built from these + C_max.
    eps_init: np.ndarray = field(default_factory=lambda: np.array([]))  # J per RPC, per owner
    kappa: np.ndarray = field(default_factory=lambda: np.array([]))     # J per feature, per owner
    d: int = 128                                                        # feature dimension
    c_max: np.ndarray = field(default_factory=lambda: np.array([]))     # feasible bulk-transfer cap (rows), per owner

    def __post_init__(self):
        P = self.num_partitions
        t = np.asarray(self.t_miss, dtype=np.float64).reshape(-1)
        if t.size == 0:
            t = np.full(P, 4.0e-4, dtype=np.float64)   # ~0.4 ms per remote fetch
        if t.size != P:
            raise ValueError(f"t_miss has length {t.size} but num_partitions={P}")
        self.t_miss = t

        # Structural coefficients (defaults order-of-magnitude consistent with the
        # papers: initiation-dominated at GNN-typical sizes -- crossover ~1000 rows).
        ei = np.asarray(self.eps_init, dtype=np.float64).reshape(-1)
        self.eps_init = np.full(P, 1.0e-3) if ei.size == 0 else ei
        kp = np.asarray(self.kappa, dtype=np.float64).reshape(-1)
        self.kappa = np.full(P, 8.0e-9) if kp.size == 0 else kp   # κ·d·n* = ε_init ⇒ n*≈1000
        cm = np.asarray(self.c_max, dtype=np.float64).reshape(-1)
        self.c_max = np.full(P, 50000.0) if cm.size == 0 else cm
        for nm, arr in (("eps_init", self.eps_init), ("kappa", self.kappa),
                        ("c_max", self.c_max)):
            if arr.size != P:
                raise ValueError(f"{nm} has length {arr.size} but num_partitions={P}")

    # --------------------------------------------- structural cost helpers (§3, §6)
    def per_owner_miss_cost(self, sigma=None):
        """Calibrated per-node miss cost c_m used for the weighted hot set (Thm B).

        Caching a node owned by m saves, per future access, the marginal fetch
        energy κ_m·d plus the congestion-scaled latency proxy t_miss_m·σ_m.  When
        σ ≡ 1 and κ·d is homogeneous this collapses to a constant ⇒ top-frequency.
        """
        P = self.num_partitions
        sigma = np.ones(P) if sigma is None else np.asarray(sigma, dtype=np.float64)
        return self.kappa * self.d + self.t_miss * sigma

    def n_star(self):
        """Payload crossover size n* = 1/θ = ε_init/(κ·d) (rows), per owner."""
        return self.eps_init / np.maximum(self.kappa * self.d, 1e-300)

    # ------------------------------------------------------------------ rebuild
    def t_rebuild(self, w):
        """Calibrated rebuild time for a window of length w (seconds).

        Accepts a scalar or an array; returns the same shape.
        T_rebuild(w) = a + b * w**c  (sublinear: hub reuse saturates uniques).
        """
        w = np.asarray(w, dtype=np.float64)
        return self.a + self.b * np.power(np.maximum(w, 0.0), self.c)

    def rebuild_term(self, w):
        """alpha * T_rebuild(w) -- the part of the rebuild on the critical path."""
        return self.alpha * self.t_rebuild(w)

    # -------------------------------------------------------- energy / reporting
    def energy(self, time_seconds):
        """E = P_bar * time  (A4)."""
        return self.p_bar * np.asarray(time_seconds, dtype=np.float64)

    # ----------------------------------------------------------------- fitting
    def fit_rebuild(self, windows: Sequence[float], times: Sequence[float]):
        """Least-squares fit of T_rebuild(w)=a+b*w**c to measured (w, time) pairs.

        Uses a coarse grid over c in (0,1) with a closed-form linear LS for (a,b)
        at each c -- no SciPy dependency, robust, and exact for the small number
        of profiling points GreenDyGNN's Alg.1 collects (one per window length).
        """
        w = np.asarray(windows, dtype=np.float64)
        y = np.asarray(times, dtype=np.float64)
        if w.size < 2:
            raise ValueError("need >=2 calibration points to fit rebuild model")
        best = None
        for c in np.linspace(0.05, 0.95, 91):
            X = np.column_stack([np.ones_like(w), np.power(w, c)])
            coef, residuals, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b = coef
            if b < 0:   # enforce the monotone-increasing, sublinear shape
                continue
            pred = a + b * np.power(w, c)
            sse = float(np.sum((pred - y) ** 2))
            if best is None or sse < best[0]:
                best = (sse, float(a), float(b), float(c))
        if best is None:
            raise RuntimeError("rebuild fit failed (no non-negative b found)")
        _, self.a, self.b, self.c = best
        return self.a, self.b, self.c

    def fit_miss_latency(self, per_owner_seconds: Dict[int, float]):
        """Set t_miss[m] from measured mean per-node miss latency per owner."""
        t = self.t_miss.copy()
        for pid, val in per_owner_seconds.items():
            if 0 <= pid < self.num_partitions:
                t[pid] = float(val)
        self.t_miss = t
        return self.t_miss

    # --------------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("t_miss", "eps_init", "kappa", "c_max"):
            d[k] = np.asarray(getattr(self, k)).tolist()
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "CostModel":
        d = dict(d)
        for k in ("t_miss", "eps_init", "kappa", "c_max"):
            if k in d:
                d[k] = np.asarray(d[k], dtype=np.float64)
        # tolerate extra keys
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def load(cls, path: str) -> "CostModel":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def default_model(num_partitions: int = 4, **overrides) -> CostModel:
    """A reasonable default calibrated model for P partitions.

    The numbers are order-of-magnitude consistent with the GreenDyGNN/GreenGNN
    measurements (sub-ms rebuilds amortised over a window, ~0.4 ms per remote
    fetch, ~190 W mean active power on a P100 node).  For paper numbers, replace
    via `CostModel.fit_*` from a real profiling run.
    """
    m = CostModel(num_partitions=num_partitions)
    for k, v in overrides.items():
        setattr(m, k, v)
    m.__post_init__()
    return m
