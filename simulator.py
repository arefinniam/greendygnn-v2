#!/usr/bin/env python3
"""GreenDyGNN v2 calibrated training simulator (V2_SPEC, Agent S items 1-2).

dgl-free, torch-free, numpy-only, fully seeded.

Cost model (paper Eq. 1-4, extended and re-anchored to live cluster measurements)
---------------------------------------------------------------------------------
T_step(W) = t_base                                   (compute + AllReduce)
          + alpha_crit * T_rebuild(W, kappa) / W     (amortised rebuild)
          + max_m S_miss(rows_m, kappa_m, delta_m)   (per-step miss straggler, Eq. 3)
          + c_ar * (max_m sigma_m - 1)               (AllReduce straggler)

TWO distinct congestion axes, matching the physical injection classes:
  * kappa_m >= 1 : per-row cost multiplier on the link to owner m — what a
    bandwidth throttle (tc tbf, class C1) does. Real-cluster anchor: netbench
    measured per-row bulk cost 0.35us/row clean -> 67us/row at 50mbit
    (kappa=192), i.e. exactly linear in the throttle ratio.
  * delta_m >= 0 : additive per-RPC delay (seconds) — what netem delay (C4)
    does; it multiplies with the number of pipelined RPC rounds, not rows.

Miss path vs rebuild path — the two are physically different and the model
keeps them separate (this distinction IS the paper's Fig. 1 point):
  * MISS path (fine-grained, initiation-dominated): live DistDGL on-demand
    fetches cost ~60us/row clean (default-DGL: 800 rows -> ~65ms/step measured)
    of which only ~0.8us is wire payload; under a bandwidth throttle the
    payload+queueing share scales with kappa at ~5.7us/row/kappa-unit, which
    reproduces the measured live sweep: 1000mbit(k=9.6) +26ms/step,
    200mbit(k=48) +170ms/step, 50mbit(k=192) +700ms/step.
      S_miss(rows, kappa, delta) = rows*(t_init_row + t_pay_row*kappa)
                                   + delta * ceil(rows/rpc_rows)/q_depth
  * REBUILD path (bulk, payload-dominated): consolidated transfer at the
    netbench bulk rate, T_rebuild(W) = a + b*W^c (clean calibrated fit) plus
    congestion inflation t_bulk_row*(kappa_m-1) on each owner's share of the
    newly fetched rows + one straggling delta round.

Hit rate h(W) follows the logistic decay (Eq. 2) multiplied by a cache-pressure
factor min(1, n_hot/U(W)), U(W) = U0*W^u_exp: the real cluster showed
ogbn-products pinned at ~45% hit because the window footprint exceeds n_hot.

Allocation model: v2 allocates cache by cost (alloc.py, capped khat). The sim
approximates its effect by tilting miss composition away from expensive owners
(cache-by-cost removes the slow owner's repeated misses) while shifting rebuild
rows toward them (their nodes are fetched once, in bulk).

Domain randomization: every episode perturbs all calibration parameters by
U(-15%,+15%) and adds 3% multiplicative observation noise, so the policy cannot
overfit the point calibration — this absorbs the residual error of the RPC fit
and makes sim-to-real transfer a trained-in robustness property.

Action space is W ONLY (8 choices). Per-owner allocation is NOT an RL action in
v2 — it is computed analytically (alloc.py) from online-estimated khat (the
marginal-greedy optimum in the payload-dominated regime). The DQN handles the
genuinely sequential part: rebuild-window selection under non-stationary
congestion.

Reward: r_t = -(E_window_per_step / E_ref(t)) * (steps_in_window/total_steps),
E_ref = per-step energy of the best static W at the CURRENT congestion. The
step-count weighting makes episode return the energy-weighted mean ratio —
comparable across policies with different decision counts (a W=1 policy makes
3000 decisions, a W=128 policy 24); an ideal policy's return is ~-1.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

STATE_SPEC_VERSION = 1
W_CHOICES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
W_NOMINAL = 16

# Feature normalisers (shared by sim + deployed controller via build_state).
KHAT_NORM = 8.0     # == alloc.DEFAULT_KHAT_CAP; khat feature saturates at cap
SIGMA_NORM = 10.0
STEP_RATIO_CLIP = 5.0

ARCHETYPES = (
    "clean", "single-link-delay", "single-link-bw",
    "two-link-sym", "two-link-asym", "oscillating", "burst-organic",
)

# true per-owner kappa measured by netbench under each tbf rate on the real
# cluster (2026-06-29 sweep; rate halves => kappa doubles)
TRUE_KAPPA_BY_RATE = {"1000mbit": 9.9, "500mbit": 19.4, "200mbit": 48.4,
                      "100mbit": 96.8, "50mbit": 193.7}
SEVERITIES = ("mild", "moderate", "severe")
SEV_KAPPA = {"mild": 5.0, "moderate": 25.0, "severe": 120.0}   # tbf-like
SEV_DELAY = {"mild": 0.002, "moderate": 0.008, "severe": 0.020}  # netem-like


# --------------------------------------------------------------------------- calib
@dataclass
class CalibParams:
    """Calibration parameter set loaded from JSON (one per dataset/cluster)."""
    # miss path (fine-grained, live anchors — see module docstring)
    t_init_row: float = 60e-6         # s/row, initiation-dominated live miss cost
    t_pay_row: float = 5.7e-6         # s/row per kappa unit (live payload+queueing)
    rpc_rows: int = 64                # rows per miss RPC (delay-round granularity)
    q_depth: int = 4                  # concurrent miss RPCs (pipelining)
    # rebuild path (bulk)
    a: float = 0.03                   # s, clean rebuild fixed cost
    b: float = 0.012                  # s
    c: float = 0.70
    t_bulk_row: float = 0.35e-6       # s/row bulk transfer (netbench clean)
    alpha_crit: float = 0.35          # fraction of rebuild on the critical path
    # hit-rate curve (Eq. 2)
    hmin: float = 0.30
    hmax: float = 0.95
    w_half: float = 32.0
    gamma_h: float = 1.5
    # workload
    t_base: float = 15e-3             # s/step (compute + clean AllReduce)
    feat_bytes: int = 2408            # per-node feature bytes (reddit: 602*f32)
    R_per_batch: float = 800.0        # expected remote nodes per batch
    batches_per_epoch: int = 100
    n_epochs: int = 30
    n_hot: int = 100000
    P: int = 4
    # footprint model (cache pressure)
    U0: float = 1200.0
    u_exp: float = 0.90
    reuse_frac: float = 0.60          # hot-set overlap between windows
    # power split (per node; scale-invariant for the policy, real for reports)
    p_cpu_w: float = 180.0
    p_gpu_active_w: float = 120.0
    p_gpu_idle_w: float = 30.0
    gpu_active_frac: float = 0.6
    # congestion coupling
    c_ar: float = 2.0e-3              # AllReduce straggler coefficient (s)
    alloc_tilt: float = 0.8           # miss-composition tilt when alloc on
    # legacy RPC-regression coefficients (Alg. 1 phase 1; kept for reference)
    alpha_rpc: float = 4.67e-3
    beta: float = 1.40e-9
    gamma_c: float = 2.01e-10
    # --- OVERLAP-AWARE COUPLING PARAMETERS (fit_coupling.py) ----------------
    # The async pipeline hides most communication under compute; the original
    # model charged full wire time and over-predicted congested cells 2-5x
    # while double-subtracting overlap collapsed t_base (clean cells under-
    # predicted 40-80%). The v2 structural form (used when t_c is not None):
    #
    #   T = t_c + alpha_reb*T_reb(W)/W
    #       + max(0,  phi_miss*sum_m M_m(W)*kappa_m
    #               + phi_reb *sum_m B_m(W)*kappa_m / W  -  beta_ov*t_c)
    #       + (phi_delay*rounds_victim(W) + r0_delay) * max_m delta_m
    #
    # with M_m = rows_m*t_pay_row (per-step miss wire), B_m = per-window
    # rebuild bulk wire rows*t_bulk_row (the 1/W scaling of this term is what
    # makes SMALL W catastrophic under bandwidth congestion — measured reddit
    # 50mbit: W4 2274ms vs W16 917ms), beta_ov*t_c the overlap budget under
    # which wire hides (measured: clean step time is W-flat on products while
    # h(W) collapses 0.75->0.05), and the delay term = exposed RPC rounds.
    # Couplings are fitted per dataset on the FIT cells only (clean W-sweep +
    # W16 severity cells); holdout cells (*_vg_*) are never touched.
    # None => legacy (pre-overlap) model, kept for synthetic development.
    t_c: Optional[float] = None       # true steady compute floor (s)
    alpha_reb: Optional[float] = None # exposed share of clean rebuild
    phi_miss: Optional[float] = None  # exposed fraction of miss wire
    phi_reb: Optional[float] = None   # exposed fraction of rebuild bulk wire
    phi_delay: Optional[float] = None # exposed fraction of miss RPC rounds
    r0_delay: Optional[float] = None  # constant exposed rounds per step
    beta_ov: Optional[float] = None   # overlap budget in units of t_c
    # --- sim-to-real OBSERVATION MODEL (calibrated) -------------------------
    # The controller does NOT observe true congestion severity: the in-band
    # per-owner signal is compressed by cache hits (few misses => few samples)
    # and by the fine-grained miss path's fixed cost. Calibrated on the real
    # cluster (fit_calibration.py): e.g. products true kappa 9.9->obs 2.9,
    # 193.7->obs 66.7; reddit is nearly blind below kappa~20 at 98% hit.
    # obs_kappa_map: [[true_kappa, observed_rtt_ratio], ...] (sorted by true)
    # obs_delay_map: [[delta_seconds, observed_fetch_ratio], ...]
    # When None, observations fall back to the physics-derived (uncompressed)
    # signal. TRUE severity always drives the DYNAMICS (step time, energy);
    # only what the agent SEES is compressed.
    obs_kappa_map: Optional[list] = None
    obs_delay_map: Optional[list] = None

    @classmethod
    def load(cls, path: str) -> "CalibParams":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, d: dict) -> "CalibParams":
        d = dict(d)
        aliases = {"P_bar_w": "p_cpu_w", "P_idle": "p_gpu_idle_w",
                   "gamma": "gamma_h", "w50": "w_half"}
        for k_alt, k in aliases.items():
            if k_alt in d and k not in d:
                d[k] = d.pop(k_alt)
        prov = d.get("_provenance") or {}
        # derive the observation maps from the calibration-run tables when not
        # given explicitly (kappa_inband/fetch_time_ratio are what the deployed
        # controller's estimators actually measured under each injection)
        if "obs_kappa_map" not in d and "kappa_table" in prov:
            m = []
            for rate, row in prov["kappa_table"].items():
                true_k = TRUE_KAPPA_BY_RATE.get(rate)
                if true_k is not None and "kappa_inband" in row:
                    m.append([float(true_k), float(row["kappa_inband"])])
            if m:
                d["obs_kappa_map"] = sorted(m)
        if "obs_delay_map" not in d and "sigma_table" in prov:
            m = []
            for ms, row in prov["sigma_table"].items():
                if "fetch_time_ratio" in row:
                    m.append([float(ms) * 1e-3, float(row["fetch_time_ratio"])])
            if m:
                d["obs_delay_map"] = sorted(m)
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


_DR_FIELDS = ("t_init_row", "t_pay_row", "a", "b", "c", "t_bulk_row", "hmin",
              "hmax", "w_half", "gamma_h", "t_base", "U0", "p_cpu_w",
              "p_gpu_active_w", "p_gpu_idle_w", "c_ar", "alloc_tilt",
              "R_per_batch", "reuse_frac")

_COUPLING_FIELDS = ("t_c", "alpha_reb", "phi_miss", "phi_reb", "phi_delay",
                    "r0_delay", "beta_ov")


def _randomized(p: CalibParams, rng: np.random.Generator,
                spread: float = 0.15) -> CalibParams:
    """Domain randomization: perturb calibration by U(-spread, +spread)."""
    q = CalibParams(**asdict(p))
    for name in _DR_FIELDS:
        setattr(q, name, getattr(q, name) * float(rng.uniform(1 - spread,
                                                              1 + spread)))
    for name in _COUPLING_FIELDS:   # fitted couplings are uncertain too
        v = getattr(q, name)
        if v is not None:
            setattr(q, name, float(v) * float(rng.uniform(1 - spread,
                                                          1 + spread)))
    # miss-RPC batching granularity is the least-identified part of the miss
    # path (it sets how strongly per-RPC delay scales with W): randomize it so
    # the policy stays robust to either regime.
    q.rpc_rows = int(rng.choice([32, 64, 128]))
    q.q_depth = int(rng.choice([2, 4, 8]))
    # preserve the CALIBRATED h(W) orientation: classic decay (hmax>hmin) AND
    # the inverted reddit curve (hit RISES with W because the async builder
    # can't keep up at small W) are both physical. Independent jitter can flip
    # a narrow curve's orientation — restore it, and enforce separation.
    inverted = p.hmin > p.hmax
    lo, hi = sorted((q.hmin, q.hmax))
    if hi - lo < 0.02:
        hi = min(0.99, lo + 0.02)
    q.hmin, q.hmax = (hi, lo) if inverted else (lo, hi)
    q.hmin = float(np.clip(q.hmin, 0.02, 0.99))
    q.hmax = float(np.clip(q.hmax, 0.02, 0.99))
    q.c = min(0.95, max(0.05, q.c))
    q.reuse_frac = min(0.95, max(0.05, q.reuse_frac))
    # a t_base at the estimator floor (<1ms) is under-identified, not truly
    # ~0: draw it from a wide range instead of scaling the floor value
    if p.t_base < 1e-3:
        q.t_base = float(rng.uniform(1e-4, 2e-3))
    # observation-model jitter: the compression maps are calibrated at W=16
    # on specific runs; perturb the observed column so the policy tolerates
    # observation-model error too
    for name in ("obs_kappa_map", "obs_delay_map"):
        m = getattr(q, name)
        if m:
            f = float(rng.uniform(1 - spread, 1 + spread))
            setattr(q, name, [[float(t), float(o) * f] for t, o in m])
    return q


# --------------------------------------------------------------------------- model
class StepModel:
    """Analytic per-step time/energy model for one (possibly randomized) param set."""

    def __init__(self, p: CalibParams, owner_shares: Optional[np.ndarray] = None,
                 alloc_on: bool = True):
        self.p = p
        self.n_remote = p.P - 1
        if owner_shares is None:
            owner_shares = np.full(self.n_remote, 1.0 / self.n_remote)
        self.owner_shares = np.asarray(owner_shares, dtype=np.float64)
        self.owner_shares = self.owner_shares / self.owner_shares.sum()
        self.alloc_on = alloc_on

    # -- primitives ---------------------------------------------------------
    def footprint(self, W: float) -> float:
        return self.p.U0 * float(W) ** self.p.u_exp

    def hit_rate(self, W: float) -> float:
        p = self.p
        h = p.hmin + (p.hmax - p.hmin) / (1.0 + (W / p.w_half) ** p.gamma_h)
        pressure = min(1.0, p.n_hot / max(1.0, self.footprint(W)))
        return float(np.clip(h * pressure, 0.0, 1.0))

    def miss_row_cost(self, kappa: float, delta_s: float = 0.0) -> float:
        """Per-row cost of the fine-grained miss path (initiation-dominated)."""
        return self.p.t_init_row + self.p.t_pay_row * max(1.0, kappa)

    def miss_stall(self, rows: float, kappa: float, delta_s: float) -> float:
        if rows <= 0:
            return 0.0
        p = self.p
        rounds = math.ceil(rows / p.rpc_rows) / max(1, p.q_depth)
        return rows * self.miss_row_cost(kappa) + delta_s * max(1.0, rounds)

    def _cost_vec(self, kappa: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Observed-cost proxy per owner (per-row, relative units) for tilting."""
        base = self.miss_row_cost(1.0)
        per_row = (self.p.t_init_row + self.p.t_pay_row * np.maximum(kappa, 1.0)
                   + delta / max(1, self.p.rpc_rows))
        return per_row / base

    def _miss_shares(self, kappa: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Per-owner miss composition; tilted away from expensive owners when
        cost-aware allocation is active (mirrors alloc.cap_khat's clamp)."""
        if not self.alloc_on:
            return self.owner_shares
        cost = np.minimum(np.maximum(self._cost_vec(kappa, delta), 1.0), KHAT_NORM)
        w = self.owner_shares * cost ** (-self.p.alloc_tilt)
        return w / w.sum()

    # -- per-step decomposition ---------------------------------------------
    def step_time(self, W: int, kappa: np.ndarray, delta: np.ndarray
                  ) -> Tuple[float, Dict[str, object]]:
        if self.p.t_c is not None:
            return self._step_time_overlap(W, kappa, delta)
        return self._step_time_legacy(W, kappa, delta)

    def _step_time_overlap(self, W: int, kappa: np.ndarray, delta: np.ndarray
                           ) -> Tuple[float, Dict[str, object]]:
        """Overlap-aware model (see CalibParams coupling docstring).

        Validated structure: (i) clean cells are near W-flat because miss wire
        hides under the beta_ov*t_c overlap budget; (ii) bandwidth congestion
        exposes wire linearly in kappa, split between the per-step miss path
        (phi_miss) and the per-window rebuild bulk path (phi_reb, ~1/W — the
        measured small-W catastrophe under tbf); (iii) delay costs exposed RPC
        rounds. All primitives (h, U, T_reb, per-row costs) come from
        fit_calibration; only the couplings are fitted by fit_coupling.py."""
        p = self.p
        kappa = np.asarray(kappa, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        h = self.hit_rate(W)
        miss_rows = p.R_per_batch * (1.0 - h)
        shares = self._miss_shares(kappa, delta)
        rows_m = np.maximum(miss_rows * shares, 0.0)

        # exposed wire beyond the overlap budget (shared NIC => sum over owners)
        M = float(np.sum(rows_m * p.t_pay_row * np.maximum(kappa, 1.0)))
        new_rows = min(self.footprint(W), float(p.n_hot)) * (1.0 - p.reuse_frac)
        if self.alloc_on:
            reb_shares = np.clip(2.0 * self.owner_shares - shares, 0.0, None)
            s = reb_shares.sum()
            reb_shares = self.owner_shares if s <= 0 else reb_shares / s
        else:
            reb_shares = self.owner_shares
        B = float(np.sum(new_rows * reb_shares * p.t_bulk_row
                         * np.maximum(kappa, 1.0)))
        wire = p.phi_miss * M + p.phi_reb * B / max(1, W)
        exposed = max(0.0, wire - p.beta_ov * p.t_c)

        # exposed per-RPC delay rounds (straggler owner)
        dmax = float(delta.max())
        d_term = 0.0
        if dmax > 0:
            v = int(np.argmax(delta))
            rounds = rows_m[v] / max(1, p.rpc_rows) / max(1, p.q_depth)
            d_term = (p.phi_delay * rounds + p.r0_delay) * dmax

        t_reb_clean = max(0.0, p.a) + max(0.0, p.b) * float(W) ** p.c
        amort = p.alpha_reb * t_reb_clean / max(1, W)

        T = p.t_c + amort + exposed + d_term
        return T, {"t_base": p.t_c, "rebuild_amort": amort,
                   "stall": exposed + d_term, "t_ar": 0.0, "hit_rate": h,
                   "miss_rows": miss_rows, "t_rebuild": t_reb_clean,
                   "rows_m": rows_m, "shares": shares}

    def _step_time_legacy(self, W: int, kappa: np.ndarray, delta: np.ndarray
                          ) -> Tuple[float, Dict[str, object]]:
        p = self.p
        kappa = np.asarray(kappa, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)
        h = self.hit_rate(W)
        miss_rows = p.R_per_batch * (1.0 - h)
        shares = self._miss_shares(kappa, delta)
        rows_m = miss_rows * shares
        stall = max(self.miss_stall(rows_m[m], kappa[m], delta[m])
                    for m in range(self.n_remote)) if miss_rows > 0.5 else 0.0

        # rebuild: clean fit + bulk congestion inflation on new rows
        new_rows = min(self.footprint(W), float(p.n_hot)) * (1.0 - p.reuse_frac)
        if self.alloc_on:
            reb_shares = np.clip(2.0 * self.owner_shares - shares, 0.0, None)
            s = reb_shares.sum()
            reb_shares = self.owner_shares if s <= 0 else reb_shares / s
        else:
            reb_shares = self.owner_shares
        t_reb = max(0.0, p.a) + max(0.0, p.b) * float(W) ** p.c
        t_reb += float(np.sum(p.t_bulk_row * np.maximum(kappa - 1.0, 0.0)
                              * new_rows * reb_shares))
        t_reb += float(delta.max())          # one straggling bulk RPC round
        amort = p.alpha_crit * t_reb / max(1, W)

        sigma_lat = 1.0 + delta / max(p.alpha_rpc, 1e-9)
        t_ar = p.c_ar * max(0.0, float(sigma_lat.max()) - 1.0)

        T = p.t_base + amort + stall + t_ar
        return T, {"t_base": p.t_base, "rebuild_amort": amort, "stall": stall,
                   "t_ar": t_ar, "hit_rate": h, "miss_rows": miss_rows,
                   "t_rebuild": t_reb, "rows_m": rows_m, "shares": shares}

    @property
    def t_floor(self) -> float:
        """Steady compute floor: t_c under the overlap model, else t_base."""
        return self.p.t_c if self.p.t_c is not None else self.p.t_base

    def step_energy(self, T: float) -> Tuple[float, float, float]:
        """(total_j, gpu_j, cpu_j) for one step of duration T (A4 split)."""
        p = self.p
        t_active = min(T, p.gpu_active_frac * self.t_floor)
        gpu_j = p.p_gpu_active_w * t_active + p.p_gpu_idle_w * (T - t_active)
        cpu_j = p.p_cpu_w * T
        return gpu_j + cpu_j, gpu_j, cpu_j

    def grid(self, kappa: np.ndarray, delta: np.ndarray,
             w_choices: Sequence[int] = W_CHOICES) -> np.ndarray:
        """T_step for each W at fixed congestion (tests/oracle/reference)."""
        return np.array([self.step_time(w, kappa, delta)[0] for w in w_choices])

    def best_static_w(self, kappa: np.ndarray, delta: np.ndarray,
                      w_choices: Sequence[int] = W_CHOICES) -> int:
        return int(np.argmin(self.grid(kappa, delta, w_choices)))


# ---------------------------------------------------------------- congestion profile
@dataclass
class Segment:
    t0: int                      # step index (inclusive)
    t1: int                      # step index (exclusive)
    kappa: np.ndarray            # (P-1,)
    delta: np.ndarray            # (P-1,) seconds


def _profile(archetype: str, severity: str, total_steps: int, n_remote: int,
             rng: np.random.Generator) -> List[Segment]:
    """Seeded congestion profile: contiguous segments covering [0, total)."""
    k1 = np.ones(n_remote)
    d0 = np.zeros(n_remote)
    kv = SEV_KAPPA[severity]
    dv = SEV_DELAY[severity]

    def seg(t0, t1, kappa=None, delta=None):
        return Segment(int(t0), int(t1),
                       k1.copy() if kappa is None else np.asarray(kappa, float),
                       d0.copy() if delta is None else np.asarray(delta, float))

    if archetype == "clean":
        return [seg(0, total_steps)]

    onset = int(rng.uniform(0.05, 0.35) * total_steps)
    dur = int(rng.uniform(0.30, 0.60) * total_steps)
    end = min(total_steps, onset + dur)
    segs: List[Segment] = []

    def sandwich(mid_kappa, mid_delta):
        if onset > 0:
            segs.append(seg(0, onset))
        segs.append(seg(onset, end, mid_kappa, mid_delta))
        if end < total_steps:
            segs.append(seg(end, total_steps))
        return segs

    victims = rng.permutation(n_remote)
    if archetype == "single-link-delay":
        d = d0.copy(); d[victims[0]] = dv * rng.uniform(0.75, 1.25)
        return sandwich(None, d)
    if archetype == "single-link-bw":
        k = k1.copy(); k[victims[0]] = kv * rng.uniform(0.75, 1.25)
        return sandwich(k, None)
    if archetype == "two-link-sym":
        if rng.random() < 0.5:
            k = k1.copy(); k[victims[:2]] = kv * rng.uniform(0.75, 1.25)
            return sandwich(k, None)
        d = d0.copy(); d[victims[:2]] = dv * rng.uniform(0.75, 1.25)
        return sandwich(None, d)
    if archetype == "two-link-asym":
        k = k1.copy()
        k[victims[0]] = kv * rng.uniform(0.75, 1.25)
        k[victims[1]] = max(1.0, kv * rng.uniform(0.15, 0.5))
        d = d0.copy()
        if rng.random() < 0.5:
            d[victims[1]] = dv * rng.uniform(0.5, 1.0)
        return sandwich(k, d)
    if archetype in ("oscillating", "burst-organic"):
        period = max(4, int(rng.uniform(0.05, 0.20) * total_steps))
        duty = rng.uniform(0.3, 0.7)
        v = victims[0]
        use_delay = archetype == "oscillating" and rng.random() < 0.5
        t = 0
        while t < total_steps:
            on_len = max(1, int(period * duty))
            off_len = max(1, period - on_len)
            if use_delay:
                d = d0.copy(); d[v] = dv * rng.uniform(0.75, 1.25)
                segs.append(seg(t, min(total_steps, t + on_len), None, d))
            else:
                k = k1.copy(); k[v] = kv * rng.uniform(0.75, 1.25)
                segs.append(seg(t, min(total_steps, t + on_len), k, None))
            t += on_len
            if t < total_steps:
                segs.append(seg(t, min(total_steps, t + off_len)))
                t += off_len
        return segs
    raise ValueError(f"unknown archetype {archetype}")


# --------------------------------------------------------------------------- state
def build_state(khat: np.ndarray, sigma: np.ndarray, miss_share: np.ndarray,
                hit_rate: float, step_ratio: float, rebuild_frac: float,
                batches_remaining_norm: float, w_prev_idx: int,
                n_w: int = len(W_CHOICES)) -> np.ndarray:
    """Canonical state vector (STATE_SPEC_VERSION=1); single source of truth
    shared by the simulator and the deployed controller.

    dim = 3*(P-1) + 4 + n_w  (P=4, n_w=8 -> 21). Every feature is computable at
    deployment from I1 fetch events + step timing (deploy-computable contract).
    """
    khat = np.clip(np.asarray(khat, np.float64), 1.0, KHAT_NORM) / KHAT_NORM
    sigma = np.clip(np.asarray(sigma, np.float64), 1.0, SIGMA_NORM) / SIGMA_NORM
    miss_share = np.clip(np.asarray(miss_share, np.float64), 0.0, 1.0)
    onehot = np.zeros(n_w)
    onehot[int(w_prev_idx)] = 1.0
    scal = np.array([
        float(np.clip(hit_rate, 0.0, 1.0)),
        float(np.clip(step_ratio, 0.0, STEP_RATIO_CLIP)) / STEP_RATIO_CLIP,
        float(np.clip(rebuild_frac, 0.0, 1.0)),
        float(np.clip(batches_remaining_norm, 0.0, 1.0)),
    ])
    return np.concatenate([khat, sigma, miss_share, scal, onehot]).astype(np.float32)


def state_dim(P: int, n_w: int = len(W_CHOICES)) -> int:
    return 3 * (P - 1) + 4 + n_w


# ----------------------------------------------------------------------- simulator
class GreenDyGNNSim:
    """Episodic simulator. One episode = one distributed training run; one
    decision per rebuild boundary; action = index into w_choices."""

    def __init__(self, calib: CalibParams, seed: int = 0,
                 domain_randomization: bool = True, obs_noise: float = 0.03,
                 dr_spread: float = 0.15, w_choices: Sequence[int] = W_CHOICES,
                 ewma_alpha: float = 0.3, alloc_on: bool = True,
                 archetypes: Sequence[str] = ARCHETYPES,
                 reward_scale: float = 20.0):
        self.calib = calib
        self.rng = np.random.default_rng(seed)
        self.dr = domain_randomization
        self.dr_spread = dr_spread
        self.obs_noise = obs_noise
        self.w_choices = tuple(int(w) for w in w_choices)
        self.ewma_alpha = ewma_alpha
        self.alloc_on = alloc_on
        self.archetypes = tuple(archetypes)
        self.n_remote = calib.P - 1
        # Reward is ADVANTAGE-STYLE and scaled (see step()): zero-centered so
        # the Q-gap between W choices is not drowned by a large common baseline,
        # then multiplied by reward_scale to lift it above Huber/optimizer
        # noise. Both transforms are affine per decision, so the optimal policy
        # is unchanged; energy comparisons for REPORTING use norm_energy, which
        # is reward-scale-free.
        self.reward_scale = float(reward_scale)
        self._episode_seed: Optional[int] = None

    # -- episode setup -------------------------------------------------------
    def reset(self, episode_seed: Optional[int] = None) -> np.ndarray:
        if episode_seed is None:
            episode_seed = int(self.rng.integers(0, 2 ** 31 - 1))
        self._episode_seed = int(episode_seed)
        erng = np.random.default_rng(self._episode_seed)
        self.erng = erng

        p = _randomized(self.calib, erng, self.dr_spread) if self.dr \
            else CalibParams(**asdict(self.calib))
        shares = erng.dirichlet(np.full(self.n_remote, 8.0))
        self.model = StepModel(p, owner_shares=shares, alloc_on=self.alloc_on)

        arch = str(erng.choice(list(self.archetypes)))
        sev = str(erng.choice(list(SEVERITIES)))
        self.total_steps = p.batches_per_epoch * p.n_epochs
        self.segments = _profile(arch, sev, self.total_steps, self.n_remote, erng)
        self.archetype, self.severity = arch, sev
        # per-episode memoization: costs are piecewise-constant per segment,
        # so (w, segment) fully determines step_time/energy for this episode
        self._seg_starts = [s.t0 for s in self.segments]
        self._te_cache: Dict[Tuple[int, int], Tuple[float, float, dict]] = {}
        self._ref_cache: Dict[int, float] = {}

        self.t = 0
        self.w_prev_idx = self.w_choices.index(W_NOMINAL) \
            if W_NOMINAL in self.w_choices else len(self.w_choices) // 2
        self._khat = np.ones(self.n_remote)
        self._sigma = np.ones(self.n_remote)
        self._miss_share = self.model.owner_shares.copy()
        self._hit = self.model.hit_rate(self.w_choices[self.w_prev_idx])
        self._step_ratio = 1.0
        self._rebuild_frac = 0.1
        self.ep_energy_j = 0.0
        self.ep_time_s = 0.0
        self.ep_reward = 0.0
        self.ep_ref_j = 0.0        # Σ E_ref(t)·n — denominator of norm_energy
        self.n_decisions = 0
        # behavioral trace: step-weighted mean W in clean vs congested spans
        self._w_clean = [0.0, 0]   # [Σ W·n, Σ n]
        self._w_cong = [0.0, 0]
        return self._obs()

    def _seg_at(self, t: int) -> int:
        import bisect
        i = bisect.bisect_right(self._seg_starts, t) - 1
        return max(0, min(i, len(self.segments) - 1))

    def congestion_at(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        s = self.segments[self._seg_at(t)]
        if s.t0 <= t < s.t1:
            return s.kappa, s.delta
        return np.ones(self.n_remote), np.zeros(self.n_remote)

    def _step_te(self, W: int, seg_idx: int) -> Tuple[float, float, dict]:
        """Memoized (T, E, comps) for window length W inside one segment —
        costs are piecewise-constant per segment for a fixed episode."""
        key = (W, seg_idx)
        hit = self._te_cache.get(key)
        if hit is not None:
            return hit
        s = self.segments[seg_idx]
        T, comps = self.model.step_time(W, s.kappa, s.delta)
        E, _, _ = self.model.step_energy(T)
        out = (T, E, comps)
        self._te_cache[key] = out
        return out

    # -- window rollout ------------------------------------------------------
    def _window_cost(self, t0: int, W: int) -> Tuple[float, float, Dict[str, object]]:
        """Average per-step (T, E) over window [t0, t0+W), honoring segment
        boundaries (congestion can change mid-window)."""
        t, remaining = t0, min(W, self.total_steps - t0)
        span = max(1, remaining)
        tot_T = tot_E = 0.0
        comps_last: Dict[str, object] = {}
        acc = {"rebuild_amort": 0.0, "stall": 0.0, "t_ar": 0.0}
        while remaining > 0:
            si = self._seg_at(t)
            nxt = self.segments[si].t1
            n = min(remaining, max(1, nxt - t))
            T, E, comps = self._step_te(W, si)
            tot_T += T * n
            tot_E += E * n
            for k in acc:
                acc[k] += float(comps[k]) * n
            comps_last = comps
            t += n
            remaining -= n
        out = dict(comps_last)
        for k, v in acc.items():
            out[k] = v / span
        return tot_T / span, tot_E / span, out

    def _ref_energy(self, t0: int) -> float:
        """Per-step energy of the best static W at the CURRENT congestion —
        the scale-invariant reward reference (paper Eq. 5's E_ref). Memoized
        per segment."""
        si = self._seg_at(t0)
        hit = self._ref_cache.get(si)
        if hit is not None:
            return hit
        best = math.inf
        for i, w in enumerate(self.w_choices):
            _, E, _ = self._step_te(w, si)
            best = min(best, E)
        self._ref_cache[si] = best
        return best

    def step(self, action_idx: int):
        W = self.w_choices[int(action_idx)]
        t0 = self.t
        n = min(W, self.total_steps - t0)
        T_ps, E_ps, comps = self._window_cost(t0, W)
        self.t = t0 + n
        self.ep_time_s += T_ps * n
        self.ep_energy_j += E_ps * n
        e_ref = self._ref_energy(t0)
        self.ep_ref_j += e_ref * n

        # advantage-style, zero-centered, scaled reward (see __init__ note):
        # r = -(E/E_ref - 1) * n/total * scale. An always-optimal policy scores
        # ~0; the scale lifts per-decision Q-gaps above optimizer noise.
        reward = -(E_ps / e_ref - 1.0) * (n / self.total_steps) \
            * self.reward_scale
        self.ep_reward += reward
        self.n_decisions += 1
        k0, d0_ = self.congestion_at(t0)
        bucket = self._w_cong if (float(k0.max()) > 1.01
                                  or float(d0_.max()) > 0) else self._w_clean
        bucket[0] += W * n
        bucket[1] += n

        kappa, delta = self.congestion_at(min(self.t, self.total_steps - 1))
        self._update_obs(W, kappa, delta, T_ps, comps)
        self.w_prev_idx = int(action_idx)

        done = self.t >= self.total_steps
        info = {"W": W, "T_per_step": T_ps, "E_per_step": E_ps,
                "kappa": kappa.copy(), "delta": delta.copy(),
                "hit_rate": comps.get("hit_rate", 0.0)}
        return self._obs(), float(reward), bool(done), info

    # -- observations --------------------------------------------------------
    def _noisy(self, x):
        if self.obs_noise <= 0:
            return x
        return x * self.erng.uniform(1 - self.obs_noise, 1 + self.obs_noise,
                                     size=np.shape(x) or None)

    @staticmethod
    def _compress(x: np.ndarray, mp: Optional[list], anchor: Tuple[float, float]
                  ) -> Optional[np.ndarray]:
        """Sim-to-real observation model: map TRUE severity to what the deployed
        in-band estimator would actually report (calibrated compression), with
        the clean anchor point prepended so an uncongested link observes ~1.0.
        np.interp saturates beyond the calibrated range (conservative)."""
        if not mp:
            return None
        xs = [anchor[0]] + [float(t) for t, _ in mp]
        ys = [anchor[1]] + [float(o) for _, o in mp]
        return np.interp(np.asarray(x, np.float64), xs, ys)

    def _update_obs(self, W, kappa, delta, T_ps, comps):
        m = self.model
        rows_m = np.maximum(np.asarray(comps.get("rows_m")), 1e-9)
        # --- what the agent SEES (observation model) ---
        khat_c = self._compress(kappa, m.p.obs_kappa_map, (1.0, 1.0))
        sigma_c = self._compress(delta, m.p.obs_delay_map, (0.0, 1.0))
        if khat_c is not None:
            khat_inst = self._noisy(khat_c)
        else:
            rtt_pr = np.array([m.miss_row_cost(kappa[i])
                               for i in range(self.n_remote)])
            rtt_pr = np.maximum(self._noisy(rtt_pr), 1e-12)
            khat_inst = rtt_pr / max(rtt_pr.min(), 1e-12)
        if sigma_c is not None:
            sigma_inst = self._noisy(sigma_c)
        else:
            rtt = np.array([m.miss_stall(rows_m[i], kappa[i], delta[i])
                            for i in range(self.n_remote)])
            rtt = np.maximum(self._noisy(rtt), 1e-12)
            base_rtt = np.array([m.miss_stall(rows_m[i], 1.0, 0.0)
                                 for i in range(self.n_remote)])
            sigma_inst = rtt / np.maximum(base_rtt, 1e-12)
        a = self.ewma_alpha
        self._khat = (1 - a) * self._khat + a * np.maximum(1.0, khat_inst)
        self._sigma = (1 - a) * self._sigma + a * np.maximum(1.0, sigma_inst)
        share = rows_m / rows_m.sum()
        self._miss_share = (1 - a) * self._miss_share + a * share
        self._hit = float(self._noisy(float(comps.get("hit_rate", self._hit))))
        self._step_ratio = float(self._noisy(T_ps / m.t_floor))
        self._rebuild_frac = float(np.clip(
            float(comps.get("rebuild_amort", 0.0)) / max(T_ps, 1e-9), 0.0, 1.0))

    def _obs(self) -> np.ndarray:
        return build_state(
            self._khat, self._sigma, self._miss_share, self._hit,
            self._step_ratio, self._rebuild_frac,
            1.0 - self.t / max(1, self.total_steps),
            self.w_prev_idx, n_w=len(self.w_choices))

    # -- policy evaluation ---------------------------------------------------
    def rollout(self, policy: Callable[[np.ndarray, "GreenDyGNNSim"], int],
                episode_seed: int) -> Dict[str, object]:
        obs = self.reset(episode_seed)
        done = False
        while not done:
            a = int(policy(obs, self))
            obs, r, done, info = self.step(a)
        return {"reward": self.ep_reward, "energy_j": self.ep_energy_j,
                "norm_energy": self.ep_energy_j / max(self.ep_ref_j, 1e-12),
                "time_s": self.ep_time_s, "decisions": self.n_decisions,
                "archetype": self.archetype, "severity": self.severity,
                "mean_w_clean": self._w_clean[0] / self._w_clean[1]
                if self._w_clean[1] else None,
                "mean_w_congested": self._w_cong[0] / self._w_cong[1]
                if self._w_cong[1] else None,
                "seed": self._episode_seed}

    def rollout_static(self, w_idx: int, episode_seed: int) -> Dict[str, object]:
        return self.rollout(lambda obs, sim: w_idx, episode_seed)

    def oracle_static(self, episode_seed: int) -> Dict[str, object]:
        """Best static-W policy for THIS episode (post-hoc per-episode oracle).

        Selection is by norm_energy — the scale-free REPORTING metric — not by
        the shaped training reward: the reward weights windows by step share
        while norm_energy weights them by reference energy, and the two can
        rank static policies differently on strongly time-varying episodes."""
        best = None
        for i in range(len(self.w_choices)):
            r = self.rollout_static(i, episode_seed)
            if best is None or r["norm_energy"] < best["norm_energy"]:
                best = dict(r, w_idx=i)
        return best


# ---------------------------------------------------------------- baseline policies
def heuristic_policy(obs: np.ndarray, sim: GreenDyGNNSim) -> int:
    """Paper Eq. 7 threshold rule on the worst-owner inferred congestion."""
    n = sim.n_remote
    khat = obs[:n] * KHAT_NORM
    sigma = obs[n:2 * n] * SIGMA_NORM
    sev = max(float(khat.max()), float(sigma.max()))
    w0 = W_NOMINAL
    if sev <= 1.25:
        w = w0
    elif sev <= 4.0:
        w = w0 // 2
    else:
        w = w0 // 4
    wc = sim.w_choices
    return min(range(len(wc)), key=lambda i: abs(wc[i] - w))


def random_policy_factory(seed: int = 0):
    rng = np.random.default_rng(seed)
    def policy(obs, sim):
        return int(rng.integers(0, len(sim.w_choices)))
    return policy
