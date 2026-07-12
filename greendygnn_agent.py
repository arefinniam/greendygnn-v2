#!/usr/bin/env python3
"""GreenDyGNN v2 runtime controller (V2_SPEC I2).

Two-tier design — RL only where the decision is genuinely sequential:
  * Rebuild window W: Double-DQN policy TRAINED OFFLINE in the calibrated
    simulator (simulator.py + train_agent.py) and deployed here as a FROZEN
    checkpoint (eps=0, no learning, no exploration during measured runs).
  * Per-owner cache allocation: NOT an RL action. The builder calls the
    analytic marginal-greedy allocator (alloc.select_owner_budgets) with the
    controller's online khat estimate — provably optimal top-k by
    khat[owner]*count in the payload-dominated regime, capped at
    alloc.DEFAULT_KHAT_CAP.

Congestion signals (replaces the retired miss-fraction heuristic, which could
not observe congestion at all — miss composition is a property of the access
pattern, not the network):
  * khat[pid]  = EWMA(per-row fetch cost of owner pid) / min over remote owners,
                 floored at 1.0 — relative bandwidth-cost of each owner (what a
                 tbf throttle moves).
  * sigma[pid] = EWMA(per-fetch RTT of owner pid) / warm-up baseline[pid],
                 floored at 1.0, baseline = 15th percentile of that owner's
                 RTTs observed during the warm-up phase (what netem delay
                 moves).
Both are computed from the I1 per-owner fetch events recorded by the cache at
actual pull time — real latencies, per owner, on the live data path.

Decision provenance is logged with every decision: "dqn" | "fallback-heuristic"
| "warmup". With no checkpoint the controller degrades to the Eq.7-style
threshold rule and says so loudly.

Deploy-side note for the trainer/prefetcher (I4): Decision.owner_budgets holds
the per-owner allocation WEIGHTS (capped khat, fastest owner == 1.0) that the
builder passes to alloc.select_owner_budgets(khat=...); the realised integer
budgets are returned by the allocator at build time (the controller cannot know
them earlier — they depend on the window's hot-set composition). None means
uniform (top-count) allocation.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from simulator import (STATE_SPEC_VERSION, W_CHOICES, W_NOMINAL, KHAT_NORM,
                       SIGMA_NORM, build_state, state_dim)
from alloc import DEFAULT_KHAT_CAP, cap_khat

WARMUP_MIN_EVENTS_PER_OWNER = 8


class QNetwork(nn.Module):
    """Double-DQN Q-network (V2_SPEC agent spec: MLP 2x256 ReLU)."""

    def __init__(self, state_dim_: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim_, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions))

    def forward(self, x):
        return self.net(x)


def save_checkpoint(path: str, q_net: QNetwork, config: dict):
    payload = {
        "state_dict": q_net.state_dict(),
        "config": dict(config),
        "state_spec_version": STATE_SPEC_VERSION,
        "w_choices": list(config.get("w_choices", W_CHOICES)),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[QNetwork, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older torch without weights_only kwarg (cluster: 2.1 has it)
        payload = torch.load(path, map_location=device)
    ver = payload.get("state_spec_version")
    if ver != STATE_SPEC_VERSION:
        raise ValueError(
            f"checkpoint state_spec_version={ver} != code {STATE_SPEC_VERSION}; "
            "retrain the agent (train_agent.py) against this code version")
    cfg = payload["config"]
    net = QNetwork(int(cfg["state_dim"]), int(cfg["num_actions"]),
                   int(cfg.get("hidden", 256)))
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, cfg


@dataclass
class Decision:
    """One rebuild-boundary decision (I2)."""
    W: int
    owner_budgets: Optional[Dict[int, float]]  # allocation WEIGHTS (capped khat); see module docstring
    khat: Dict[int, float]
    sigma_hat: Dict[int, float]
    provenance: str                            # "dqn" | "fallback-heuristic" | "warmup"
    extras: dict = field(default_factory=dict)

    def to_log(self) -> dict:
        return {"W": int(self.W), "provenance": self.provenance,
                "khat": {int(k): round(float(v), 4) for k, v in self.khat.items()},
                "sigma_hat": {int(k): round(float(v), 4)
                              for k, v in self.sigma_hat.items()},
                "owner_budgets": None if self.owner_budgets is None else
                {int(k): round(float(v), 4) for k, v in self.owner_budgets.items()},
                **{k: v for k, v in self.extras.items()
                   if isinstance(v, (int, float, str, bool))}}


class GreenDyGNNController:
    """V2_SPEC I2 controller. Deploy mode: frozen policy, zero learning."""

    def __init__(self, num_partitions: int, local_pid: int,
                 checkpoint_path: Optional[str] = None, mode: str = "deploy",
                 seed: int = 0, w_choices: Sequence[int] = W_CHOICES,
                 device: str = "cpu", khat_cap: float = DEFAULT_KHAT_CAP,
                 ewma_alpha: float = 0.3, uniform_alloc: bool = False):
        if mode == "online":
            raise NotImplementedError(
                "online learning during measured runs is retired in v2 "
                "(it made reported numbers contain exploration noise); train "
                "offline with train_agent.py and pass checkpoint_path. "
                "Ablations use --no_rl / --uniform_alloc instead.")
        if mode != "deploy":
            raise ValueError(f"unknown mode {mode!r}")
        self.P = int(num_partitions)
        self.local_pid = int(local_pid)
        self.remote_pids: List[int] = [p for p in range(self.P)
                                       if p != self.local_pid]
        self.w_choices = tuple(int(w) for w in w_choices)
        self.device = device
        self.khat_cap = float(khat_cap)
        self.ewma_alpha = float(ewma_alpha)
        self.uniform_alloc = bool(uniform_alloc)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.q_net: Optional[QNetwork] = None
        if checkpoint_path:
            self.q_net, cfg = load_checkpoint(checkpoint_path, device)
            if tuple(cfg.get("w_choices", self.w_choices)) != self.w_choices:
                raise ValueError("checkpoint w_choices mismatch")
            exp_dim = state_dim(self.P, len(self.w_choices))
            if int(cfg["state_dim"]) != exp_dim:
                raise ValueError(
                    f"checkpoint state_dim {cfg['state_dim']} != {exp_dim} "
                    f"for P={self.P}")
        else:
            print("[greendygnn-controller] WARNING: no checkpoint provided — "
                  "running the HEURISTIC FALLBACK policy (Eq.7 threshold), "
                  "not the learned DQN.", file=sys.stderr, flush=True)

        # --- estimator state (per remote owner) ---
        self._khat_raw: Dict[int, float] = {}      # EWMA per-row cost (s/row)
        self._rtt_ewma: Dict[int, float] = {}      # EWMA per-fetch rtt (s)
        self._warmup_rtts: Dict[int, List[float]] = {p: [] for p in self.remote_pids}
        self._baseline: Dict[int, float] = {}      # per-owner warm-up p15 rtt
        self._miss_rows_ewma: Dict[int, float] = {p: 1.0 for p in self.remote_pids}
        self.in_warmup = True

        self._hit_rate = 0.5
        self._step_ratio = 1.0
        self._rebuild_frac = 0.1
        self._batches_remaining = 1.0
        self._w_prev = W_NOMINAL if W_NOMINAL in self.w_choices \
            else self.w_choices[len(self.w_choices) // 2]

        self.decision_log: List[dict] = []
        self._decide_times_ms: List[float] = []

    # ------------------------------------------------------------------ observe
    def observe(self, owner_stats: Dict[int, tuple], hit_rate: float,
                step_time: float, base_step_time: float,
                energy_j: Optional[float] = None,
                rebuild_frac: Optional[float] = None,
                batches_remaining_norm: Optional[float] = None):
        """Ingest one boundary's worth of signals.

        owner_stats: the I1 cache.get_owner_latency_stats() format — either
        {pid: {"n":, "rows":, "bytes":, "mean_rtt":, "mean_rtt_per_row":}} (the
        live cache emits dicts) or the tuple form
        {pid: (n, rows, bytes, mean_rtt, mean_rtt_per_row)}. Missing owners (no
        fetch this window) simply keep their previous EWMA — pid-KEYED, so a
        zero-miss owner can never shift another owner's slot (the v1 bug).
        """
        a = self.ewma_alpha
        for pid, st in owner_stats.items():
            pid = int(pid)
            if pid == self.local_pid or pid not in self.remote_pids:
                continue
            if isinstance(st, dict):
                n, rows, _bytes = st["n"], st["rows"], st["bytes"]
                mean_rtt = st["mean_rtt"]
                rtt_per_row = st["mean_rtt_per_row"]
            else:
                n, rows, _bytes, mean_rtt, rtt_per_row = st
            if not n or rows <= 0 or mean_rtt <= 0 or rtt_per_row <= 0:
                continue
            self._khat_raw[pid] = rtt_per_row if pid not in self._khat_raw else \
                (1 - a) * self._khat_raw[pid] + a * rtt_per_row
            self._rtt_ewma[pid] = mean_rtt if pid not in self._rtt_ewma else \
                (1 - a) * self._rtt_ewma[pid] + a * mean_rtt
            self._miss_rows_ewma[pid] = rows if pid not in self._miss_rows_ewma \
                else (1 - a) * self._miss_rows_ewma[pid] + a * rows
            if self.in_warmup:
                self._warmup_rtts[pid].append(float(mean_rtt))

        if hit_rate is not None and hit_rate >= 0:
            hr = hit_rate / 100.0 if hit_rate > 1.0 else hit_rate
            self._hit_rate = (1 - a) * self._hit_rate + a * float(hr)
        if step_time and base_step_time and base_step_time > 0:
            self._step_ratio = float(step_time / base_step_time)
        if rebuild_frac is not None:
            self._rebuild_frac = float(rebuild_frac)
        if batches_remaining_norm is not None:
            self._batches_remaining = float(batches_remaining_norm)

    def finish_warmup(self):
        """Freeze per-owner baselines (15th pct of warm-up RTTs). Called by the
        trainer at the end of the clean warm-up epochs; also auto-triggers when
        every owner has enough events."""
        for pid in self.remote_pids:
            rtts = self._warmup_rtts[pid]
            if rtts:
                self._baseline[pid] = float(np.percentile(rtts, 15))
        self.in_warmup = False

    def _maybe_auto_warmup_exit(self):
        if self.in_warmup and all(
                len(self._warmup_rtts[p]) >= WARMUP_MIN_EVENTS_PER_OWNER
                for p in self.remote_pids):
            self.finish_warmup()

    # --------------------------------------------------------------- estimators
    def khat(self) -> Dict[int, float]:
        """Per-owner relative per-row cost, fastest remote owner == 1.0."""
        if not self._khat_raw:
            return {p: 1.0 for p in self.remote_pids}
        lo = min(self._khat_raw.values())
        out = {}
        for p in self.remote_pids:
            v = self._khat_raw.get(p, lo)
            out[p] = max(1.0, v / max(lo, 1e-12))
        return out

    def sigma_hat(self) -> Dict[int, float]:
        out = {}
        for p in self.remote_pids:
            rtt = self._rtt_ewma.get(p)
            base = self._baseline.get(p)
            out[p] = max(1.0, rtt / base) if (rtt and base and base > 0) else 1.0
        return out

    def _miss_share(self) -> np.ndarray:
        v = np.array([max(0.0, self._miss_rows_ewma.get(p, 0.0))
                      for p in self.remote_pids], dtype=np.float64)
        s = v.sum()
        return v / s if s > 0 else np.full(len(self.remote_pids),
                                           1.0 / max(1, len(self.remote_pids)))

    # ------------------------------------------------------------------- state
    def state_vector(self) -> np.ndarray:
        kh = self.khat()
        sg = self.sigma_hat()
        khat_arr = np.array([kh[p] for p in self.remote_pids])
        sigma_arr = np.array([sg[p] for p in self.remote_pids])
        return build_state(khat_arr, sigma_arr, self._miss_share(),
                           self._hit_rate, self._step_ratio,
                           self._rebuild_frac, self._batches_remaining,
                           self.w_choices.index(self._w_prev),
                           n_w=len(self.w_choices))

    # ------------------------------------------------------------------ decide
    def _heuristic_w(self, kh: Dict[int, float], sg: Dict[int, float]) -> int:
        sev = max(max(kh.values(), default=1.0), max(sg.values(), default=1.0))
        w0 = W_NOMINAL
        if sev <= 1.25:
            w = w0
        elif sev <= 4.0:
            w = w0 // 2
        else:
            w = w0 // 4
        return min(self.w_choices, key=lambda x: abs(x - w))

    def decide(self) -> Decision:
        t0 = time.perf_counter()
        self._maybe_auto_warmup_exit()
        kh = self.khat()
        sg = self.sigma_hat()

        if self.in_warmup:
            W, prov = self._w_prev, "warmup"
        elif self.q_net is not None:
            s = torch.from_numpy(self.state_vector()).unsqueeze(0)
            with torch.no_grad():
                q = self.q_net(s)[0]
            W, prov = self.w_choices[int(torch.argmax(q).item())], "dqn"
        else:
            W, prov = self._heuristic_w(kh, sg), "fallback-heuristic"

        # Allocation keys on sigma_hat (per-owner mean-RTT ratio vs that owner's
        # OWN warm-up baseline), not on the cross-owner per-row ratio khat:
        # rtt/row mixes initiation into the slope for owners with small fetches
        # (live 2026-07-07: owner1 read 6-9x with the throttle on owner3), while
        # the self-normalized ratio isolates actual link degradation — the same
        # signal the calibration observation-model was built on.
        weights = None
        if not self.uniform_alloc and not self.in_warmup:
            capped = cap_khat(sg, self.khat_cap)
            if capped and max(capped.values()) > 1.05:
                weights = capped

        self._w_prev = int(W)
        d = Decision(W=int(W), owner_budgets=weights, khat=kh, sigma_hat=sg,
                     provenance=prov,
                     extras={"hit_rate": round(self._hit_rate, 4),
                             "step_ratio": round(self._step_ratio, 3)})
        ms = (time.perf_counter() - t0) * 1e3
        self._decide_times_ms.append(ms)
        self.decision_log.append(dict(d.to_log(), decide_ms=round(ms, 3)))
        return d

    # ------------------------------------------------------------------- stats
    def drain_decision_log(self) -> List[dict]:
        out, self.decision_log = self.decision_log, []
        return out

    def overhead_ms(self) -> dict:
        t = self._decide_times_ms
        if not t:
            return {"n": 0}
        return {"n": len(t), "mean_ms": float(np.mean(t)),
                "p99_ms": float(np.percentile(t, 99))}
