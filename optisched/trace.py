#!/usr/bin/env python3
"""Deterministic per-batch remote-access trace (proposal A1).

A `Trace` records, for each mini-batch b of an epoch, the set of *remote*
input (receptive-field) nodes and the owner partition of each.  Because the
sampler is seeded, this set is known before training (RapidGNN Prop. 3.1), which
is exactly what makes the offline scheduling problem an exact optimisation.

Storage is a flat CSR-style layout (one concatenated id array + per-batch
offsets) so a whole epoch trace is a couple of contiguous arrays on disk.

Two producers:
  * `Trace.from_batches`     -- wrap arrays already in memory.
  * `dump_trace.py`          -- extract a real trace from the DGL sampler.
  * `SyntheticTrace.generate`-- a controllable trace (power-law hot set, temporal
                                drift, owner skew) for local validation and for
                                exercising the gate without the cluster.

Within one batch a node appears at most once (input_nodes is a unique set), so
the within-interval frequency of a node equals the number of batches in the
interval that contain it -- the property the interval-cost identity relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Trace:
    """A single epoch's remote-access trace in CSR layout.

    Attributes
    ----------
    nodes   : int64[total]   concatenated remote node global ids over all batches
    owners  : int32[total]   owner partition id aligned with `nodes`
    offsets : int64[B+1]     batch b occupies nodes[offsets[b]:offsets[b+1]]
    num_partitions : P
    local_rank     : owner partition of this worker (excluded from remote sets)
    """

    nodes: np.ndarray
    owners: np.ndarray
    offsets: np.ndarray
    num_partitions: int
    local_rank: int = 0

    @property
    def num_batches(self) -> int:
        return int(self.offsets.size - 1)

    def batch(self, b: int):
        """(nodes, owners) for batch b (0-indexed)."""
        s, e = int(self.offsets[b]), int(self.offsets[b + 1])
        return self.nodes[s:e], self.owners[s:e]

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_batches(cls, batch_nodes: Sequence[np.ndarray],
                     batch_owners: Sequence[np.ndarray],
                     num_partitions: int, local_rank: int = 0) -> "Trace":
        offsets = np.zeros(len(batch_nodes) + 1, dtype=np.int64)
        for i, nb in enumerate(batch_nodes):
            offsets[i + 1] = offsets[i] + len(nb)
        nodes = (np.concatenate([np.asarray(n, dtype=np.int64) for n in batch_nodes])
                 if batch_nodes else np.empty(0, dtype=np.int64))
        owners = (np.concatenate([np.asarray(o, dtype=np.int32) for o in batch_owners])
                  if batch_owners else np.empty(0, dtype=np.int32))
        return cls(nodes, owners, offsets, num_partitions, local_rank)

    # ----------------------------------------------------------- (de)serialise
    def save(self, path: str):
        np.savez_compressed(
            path, nodes=self.nodes, owners=self.owners, offsets=self.offsets,
            num_partitions=np.int64(self.num_partitions),
            local_rank=np.int64(self.local_rank))

    @classmethod
    def load(cls, path: str) -> "Trace":
        z = np.load(path)
        return cls(z["nodes"].astype(np.int64), z["owners"].astype(np.int32),
                   z["offsets"].astype(np.int64), int(z["num_partitions"]),
                   int(z["local_rank"]))

    def restrict_owner(self, m: int) -> "Trace":
        """Sub-trace keeping only nodes owned by partition m (for owner-decoupled
        scheduling, Theorem G).  Same batch count; other owners removed."""
        bn, bo = [], []
        for b in range(self.num_batches):
            nodes, owners = self.batch(b)
            keep = owners == m
            bn.append(nodes[keep])
            bo.append(owners[keep])
        return Trace.from_batches(bn, bo, self.num_partitions, self.local_rank)

    # ----------------------------------------------------------- diagnostics
    def per_owner_access_counts(self) -> np.ndarray:
        """Total (node,batch) incidences per owner over the whole epoch."""
        out = np.zeros(self.num_partitions, dtype=np.int64)
        if self.owners.size:
            np.add.at(out, self.owners, 1)
        return out

    def mean_receptive_remote(self) -> float:
        if self.num_batches == 0:
            return 0.0
        return float(self.nodes.size / self.num_batches)


class SyntheticTrace:
    """Controllable synthetic trace generator for validation and the gate.

    Models the structure the papers measured:
      * a power-law remote hot set (RapidGNN: 45% accessed once, heavy tail),
      * temporal heterogeneity: the hot set *drifts* across the epoch, so the
        locally-optimal window length varies -- the regime in which non-uniform
        scheduling is supposed to help (proposal 8, heterogeneity correlation),
      * owner skew: nodes are assigned to owners with a configurable imbalance.

    `heterogeneity` in [0,1] controls how fast the hot set drifts; 0 gives a
    stationary trace (one global W is near-optimal -> small DP gain), 1 gives a
    strongly non-stationary trace (large DP gain expected).
    """

    @staticmethod
    def generate(num_batches: int = 120,
                 num_partitions: int = 4,
                 local_rank: int = 0,
                 universe: int = 20000,
                 remote_per_batch: int = 400,
                 zipf_s: float = 1.2,
                 heterogeneity: float = 0.5,
                 owner_skew: float = 0.0,
                 owner_correlation: float = 0.0,
                 seed: int = 0) -> Trace:
        """Generate one epoch's trace.

        owner_correlation in [0,1] aligns owners with the temporal drift: at 1.0
        the hot set that is active early in the epoch is owned by one partition
        and the late hot set by another, so per-owner congestion penalises
        specific temporal regions -- the mechanism by which congestion
        manufactures heterogeneity (proposal §8).  At 0.0 owners are i.i.d.
        """
        rng = np.random.default_rng(seed)
        owners_remote = [p for p in range(num_partitions) if p != local_rank]
        P = len(owners_remote)

        # The hot set is a window over a shuffled permutation that slides across
        # the epoch; heterogeneity sets the slide speed.
        perm = rng.permutation(universe)

        # Fixed owner assignment per universe node (static partition, A2).
        if owner_skew <= 0:
            rand_owner_k = rng.integers(0, P, size=universe)
        else:
            w = np.array([(1.0 + owner_skew) ** k for k in range(P)])
            w = w / w.sum()
            rand_owner_k = rng.choice(P, size=universe, p=w)
        # band owner by position in the drift permutation (temporal correlation)
        band_owner_k = np.empty(universe, dtype=np.int64)
        band_owner_k[perm] = np.minimum((np.arange(universe) * P) // universe, P - 1)
        mix = rng.random(universe) < float(owner_correlation)
        owner_k = np.where(mix, band_owner_k, rand_owner_k)
        node_owner = np.array([owners_remote[int(k)] for k in owner_k], dtype=np.int32)

        # Zipf weights over the universe (the hot set).
        ranks = np.arange(1, universe + 1)
        base_w = 1.0 / np.power(ranks, zipf_s)
        base_w /= base_w.sum()

        batch_nodes: List[np.ndarray] = []
        batch_owners: List[np.ndarray] = []
        span = max(1, int(universe * (1.0 - 0.6 * heterogeneity)))
        max_start = max(1, universe - span)
        for b in range(num_batches):
            frac = b / max(1, num_batches - 1)
            start = int(frac * heterogeneity * max_start)
            active = perm[start:start + span]
            w = base_w[:active.size].copy()
            w /= w.sum()
            k = min(remote_per_batch, active.size)
            chosen = rng.choice(active, size=k, replace=False, p=w)
            chosen = np.unique(chosen).astype(np.int64)
            batch_nodes.append(chosen)
            batch_owners.append(node_owner[chosen])

        return Trace.from_batches(batch_nodes, batch_owners,
                                  num_partitions, local_rank)
