#!/usr/bin/env python3
"""Offline Double-DQN training for the GreenDyGNN v2 window policy.

Trains ENTIRELY in the calibrated simulator (simulator.py) under
domain-randomized congestion, then ships a frozen checkpoint that the runtime
controller (greendygnn_agent.GreenDyGNNController) loads read-only. No learning
ever happens during measured cluster runs.

Spec (V2_SPEC Agent S items 3-4): Double DQN, MLP 2x256 ReLU, Huber loss,
gamma=0.99, replay 50k, batch 64, hard target sync every 100 gradient steps,
eps 1.0 -> 0.05 linearly over the first 30% of episodes, Adam lr 1e-4, grad
clip 10, fully seeded. Action = W in {1,2,4,8,16,32,64,128} only.

Evaluation: held-out seeded episode set vs
  * per-episode best-static-W oracle (upper reference for static policies;
    an adaptive policy CAN beat it on time-varying episodes),
  * the Eq.7 heuristic threshold rule,
  * uniform-random W.
Target: policy mean reward >= 98% of oracle mean reward (rewards are negative;
ratio computed as oracle/policy since closer-to-zero is better).

Usage:
  python3 train_agent.py --calib data/calib_synthetic.json \
      --out checkpoints/greendygnn_v2_synthetic.pt --episodes 6000 --seed 0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from simulator import (CalibParams, GreenDyGNNSim, W_CHOICES, state_dim,
                       heuristic_policy, random_policy_factory,
                       STATE_SPEC_VERSION)
from greendygnn_agent import QNetwork, save_checkpoint


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Replay:
    def __init__(self, capacity: int, seed: int):
        self.buf = collections.deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, *tr):
        self.buf.append(tr)

    def sample(self, n):
        batch = self.rng.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.from_numpy(np.stack(s)), torch.tensor(a),
                torch.tensor(r, dtype=torch.float32),
                torch.from_numpy(np.stack(s2)),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.buf)


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # --calib accepts a comma-separated list; episodes sample datasets
    # round-robin (uniform), per spec ("episode selects uniformly among
    # datasets"). All calibs must share P.
    calib_paths = [p.strip() for p in args.calib.split(",") if p.strip()]
    calibs = [CalibParams.load(p) for p in calib_paths]
    assert len({c.P for c in calibs}) == 1, "all calibs must share P"
    sims = [GreenDyGNNSim(c, seed=args.seed + 101 * i,
                          domain_randomization=True, obs_noise=args.obs_noise)
            for i, c in enumerate(calibs)]
    calib = calibs[0]
    sdim = state_dim(calib.P, len(W_CHOICES))
    n_act = len(W_CHOICES)

    q = QNetwork(sdim, n_act)
    tgt = QNetwork(sdim, n_act)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.buffer, args.seed + 1)

    eps_end_ep = max(1, int(0.3 * args.episodes))
    grad_steps = 0
    env_steps = 0
    rewards_hist = []
    t_start = time.time()

    for ep in range(args.episodes):
        sim = sims[ep % len(sims)]
        eps = max(0.05, 1.0 - 0.95 * ep / eps_end_ep)
        obs = sim.reset()          # seeded stream from sim.rng
        done = False
        ep_r = 0.0
        while not done:
            if random.random() < eps:
                a = random.randrange(n_act)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(q(torch.from_numpy(obs).unsqueeze(0))))
            obs2, r, done, _ = sim.step(a)
            replay.push(obs, a, r, obs2, float(done))
            obs = obs2
            ep_r += r
            env_steps += 1
            if len(replay) >= max(args.batch, args.warmup_transitions) \
                    and env_steps % args.update_every == 0:
                # one gradient step per update_every env steps (standard DQN)
                s, a_b, r_b, s2, d_b = replay.sample(args.batch)
                q_sa = q(s).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    a2 = q(s2).argmax(dim=1)
                    q2 = tgt(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
                    y = r_b + args.gamma * q2 * (1 - d_b)
                loss = nn.functional.huber_loss(q_sa, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                grad_steps += 1
                if grad_steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())
        rewards_hist.append(ep_r)
        if (ep + 1) % max(1, args.episodes // 20) == 0:
            recent = float(np.mean(rewards_hist[-200:]))
            print(f"ep {ep+1}/{args.episodes} eps={eps:.2f} "
                  f"mean_r(200)={recent:.4f} grad_steps={grad_steps} "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    train_s = time.time() - t_start
    return {"q": q, "rewards": rewards_hist, "train_s": train_s,
            "grad_steps": grad_steps, "sdim": sdim, "n_act": n_act,
            "calib": calib, "calibs": calibs, "calib_paths": calib_paths}


def dqn_policy_factory(q: QNetwork):
    def policy(obs, sim):
        with torch.no_grad():
            return int(torch.argmax(q(torch.from_numpy(obs).unsqueeze(0))))
    return policy


def evaluate(q: QNetwork, calib: CalibParams, n_eval: int, seed: int,
             obs_noise: float) -> dict:
    """Held-out evaluation: same episode seeds for every policy (paired)."""
    sim = GreenDyGNNSim(calib, seed=seed, domain_randomization=True,
                        obs_noise=obs_noise)
    ep_seeds = [int(s) for s in
                np.random.default_rng(seed + 12345).integers(0, 2**31 - 1,
                                                             size=n_eval)]
    pols = {"dqn": dqn_policy_factory(q),
            "heuristic": heuristic_policy,
            "random": random_policy_factory(seed + 7)}
    res = {name: [] for name in pols}          # rewards
    nrg = {name: [] for name in pols}          # norm_energy (scale-free)
    res["oracle_static"], nrg["oracle_static"] = [], []
    per_arch = collections.defaultdict(lambda: collections.defaultdict(list))
    for es in ep_seeds:
        o = sim.oracle_static(es)
        arch = f'{o["archetype"]}/{o["severity"]}'
        res["oracle_static"].append(o["reward"])
        nrg["oracle_static"].append(o["norm_energy"])
        per_arch[arch]["oracle_static"].append(o["norm_energy"])
        for name, pol in pols.items():
            r = sim.rollout(pol, es)
            res[name].append(r["reward"])
            nrg[name].append(r["norm_energy"])
            per_arch[arch][name].append(r["norm_energy"])
            if name == "dqn":       # behavioral trace: W response direction
                if r["mean_w_clean"] is not None:
                    per_arch[arch]["dqn_mean_w_clean"].append(r["mean_w_clean"])
                if r["mean_w_congested"] is not None:
                    per_arch[arch]["dqn_mean_w_congested"].append(
                        r["mean_w_congested"])

    def mstd(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    summary = {name: {"reward": mstd(res[name]), "norm_energy": mstd(nrg[name])}
               for name in res}
    # norm_energy >= ~1, lower is better; fraction-of-oracle = oracle/policy
    summary["policy_vs_oracle"] = float(np.mean(nrg["oracle_static"])
                                        / np.mean(nrg["dqn"]))
    summary["dqn_vs_heuristic"] = float(np.mean(nrg["heuristic"])
                                        / np.mean(nrg["dqn"]))
    summary["episodes"] = n_eval
    # per-archetype table in norm_energy (the interpretable metric)
    summary["per_archetype_norm_energy"] = {
        arch: {name: mstd(v) for name, v in d.items()}
        for arch, d in sorted(per_arch.items())}
    return summary


def main():
    # tiny MLP + batch 64: thread fan-out costs more than it saves (measured
    # 44 min CPU-time for 9.7 min wall on the multi-threaded probe)
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default="checkpoints/greendygnn_v2.pt")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=50000)
    ap.add_argument("--target-sync", type=int, default=100)
    ap.add_argument("--update-every", type=int, default=1)
    ap.add_argument("--warmup-transitions", type=int, default=1000)
    ap.add_argument("--obs-noise", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    r = train(args)
    q = r["q"]

    cfg = {"state_dim": r["sdim"], "num_actions": r["n_act"], "hidden": 256,
           "w_choices": list(W_CHOICES),
           "calib_hash": ",".join(sha256_file(p) for p in r["calib_paths"]),
           "calib_path": ",".join(os.path.abspath(p) for p in r["calib_paths"]),
           "seed": args.seed,
           "episodes": args.episodes, "gamma": args.gamma, "lr": args.lr,
           "state_spec_version": STATE_SPEC_VERSION,
           "grad_steps": r["grad_steps"], "train_seconds": round(r["train_s"], 1)}
    save_checkpoint(args.out, q, cfg)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"checkpoint saved: {args.out} ({size_kb:.0f} KB)")

    # reward curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rew = np.array(r["rewards"])
        k = max(1, len(rew) // 200)
        smooth = np.convolve(rew, np.ones(k) / k, mode="valid")
        plt.figure(figsize=(7, 3.2))
        plt.plot(rew, alpha=0.25, lw=0.5, label="episode reward")
        plt.plot(np.arange(len(smooth)) + k - 1, smooth, lw=1.5,
                 label=f"moving avg ({k})")
        plt.xlabel("episode"); plt.ylabel("return"); plt.legend()
        plt.tight_layout()
        png = os.path.join(out_dir, "reward_curve.png")
        plt.savefig(png, dpi=130)
        print(f"reward curve: {png}")
    except Exception as e:  # matplotlib genuinely optional
        print(f"(reward-curve plot skipped: {e})")

    print("evaluating on held-out episodes ...", flush=True)
    ev_all = {"checkpoint": os.path.abspath(args.out), "config": cfg,
              "per_calib": {}}
    for path, calib in zip(r["calib_paths"], r["calibs"]):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"--- eval on {name} ---", flush=True)
        ev = evaluate(q, calib, args.eval_episodes, args.seed + 999,
                      args.obs_noise)
        ev_all["per_calib"][name] = ev
        print(json.dumps({k: v for k, v in ev.items()
                          if k in ("policy_vs_oracle", "dqn_vs_heuristic")},
                         indent=2))
        ok = ev["policy_vs_oracle"] >= 0.98
        print(f"[{name}] TARGET policy>=98% of oracle (norm energy): "
              f"{'MET' if ok else 'NOT MET'} ({ev['policy_vs_oracle']*100:.1f}%)")
        print(f"[{name}] DQN vs heuristic (>1 = DQN better): "
              f"{ev['dqn_vs_heuristic']:.4f}")
        print(json.dumps(ev["per_archetype_norm_energy"], indent=1))
    rep = os.path.join(out_dir, "eval_report.json")
    with open(rep, "w") as f:
        json.dump(ev_all, f, indent=2)
    print(f"eval report: {rep}")


if __name__ == "__main__":
    main()
