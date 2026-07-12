#!/usr/bin/env python3
"""Single-process REAL trace generator (cluster-side, needs dgl + ogb).

Produces the exact object the gate consumes -- per-batch remote node sets with
owners -- from the REAL graph + a REAL METIS k-partition + REAL seeded multi-hop
neighbor sampling, WITHOUT the full DistDGL server orchestration.  Worker 0 owns
METIS part 0; its training seeds are the train nodes in part 0; remote nodes are
the sampled input nodes owned by other parts.

This is faithful to A1 (deterministic seeded sampling) and yields the same Trace
that dump_trace.py would under DistGraph, at a fraction of the operational cost.

Usage (on a node with dgl+ogb):
  python3 gen_real_trace.py --dataset reddit --parts 4 --epochs 8 \
      --batch_size 2000 --fan_out 10,25 --out traces --seed 1
"""
import argparse, os, numpy as np, torch as th, dgl


def load_graph(name):
    if name == "reddit":
        from dgl.data import RedditDataset
        g = RedditDataset(self_loop=True)[0]
        return g, "label", "train_mask"
    if name in ("ogbn-products", "ogbn-papers100M"):
        from ogb.nodeproppred import DglNodePropPredDataset
        ds = DglNodePropPredDataset(name=name)
        g, labels = ds[0]
        g.ndata["label"] = labels.view(-1)
        split = ds.get_idx_split()
        tm = th.zeros(g.num_nodes(), dtype=th.bool); tm[split["train"]] = True
        g.ndata["train_mask"] = tm
        return g, "label", "train_mask"
    raise ValueError(name)


def main(a):
    th.manual_seed(a.seed); np.random.seed(a.seed); dgl.seed(a.seed)
    g, _, tmask = load_graph(a.dataset)
    g = dgl.remove_self_loop(g); g = dgl.add_self_loop(g)
    print(f"{a.dataset}: |V|={g.num_nodes()} |E|={g.num_edges()}")

    # REAL METIS k-way partition -> per-node owner.
    try:
        owner = dgl.metis_partition_assignment(g, a.parts).numpy().astype(np.int32)
        print(f"METIS {a.parts}-way: part sizes {np.bincount(owner).tolist()}")
    except Exception as e:
        print(f"[warn] METIS unavailable ({e}); falling back to hash partition")
        owner = (np.arange(g.num_nodes()) % a.parts).astype(np.int32)

    local = 0
    train = th.nonzero(g.ndata[tmask], as_tuple=True)[0].numpy()
    seeds = train[owner[train] == local]          # worker-0 trains on its own part
    seeds = th.tensor(seeds, dtype=th.long)
    print(f"worker {local}: {seeds.numel()} train seeds (of {train.size} total)")

    owner_t = th.tensor(owner)
    sampler = dgl.dataloading.NeighborSampler([int(x) for x in a.fan_out.split(",")])
    out_dir = os.path.join(a.out, a.dataset); os.makedirs(out_dir, exist_ok=True)

    for ep in range(a.epochs):
        dl = dgl.dataloading.DataLoader(g, seeds, sampler, batch_size=a.batch_size,
                                        shuffle=True, drop_last=False)
        bn, bo = [], []
        for input_nodes, _out, _blocks in dl:
            inp = input_nodes
            ow = owner_t[inp]
            rem = ow != local
            bn.append(inp[rem].numpy().astype(np.int64))
            bo.append(ow[rem].numpy().astype(np.int32))
        # write CSR npz matching optisched.trace.Trace.load
        offs = np.zeros(len(bn) + 1, dtype=np.int64)
        for i, x in enumerate(bn):
            offs[i + 1] = offs[i] + len(x)
        nodes = np.concatenate(bn) if bn else np.empty(0, np.int64)
        owners = np.concatenate(bo) if bo else np.empty(0, np.int32)
        path = os.path.join(out_dir, f"r{local}_epoch_{ep:03d}.npz")
        np.savez_compressed(path, nodes=nodes, owners=owners, offsets=offs,
                            num_partitions=np.int64(a.parts), local_rank=np.int64(local))
        print(f"epoch {ep}: {len(bn)} batches, mean remote/batch="
              f"{nodes.size/max(1,len(bn)):.0f} -> {path}")
    print("done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="reddit")
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--fan_out", default="10,25")
    p.add_argument("--out", default="traces")
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
