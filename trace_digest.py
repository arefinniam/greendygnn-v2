"""Rolling digest of the sampled remote-access stream (trace-equivalence
blocker, RESEARCH_PLAN_v2 item 3).

Both the live BackgroundSampler (--trace_digest) and collect_trace.py fold
every batch through the same function; equal final digests prove the collector
saw the byte-identical stream a training run consumes — turning "trace-exact"
from an assertion into a checkable property.

The digest covers (global_batch_index, remote node ids in sampler order) as
little-endian int64 — BEFORE any dtype downcast, so storage format cannot mask
a mismatch.
"""

import hashlib
import json


def new_digest():
    return hashlib.md5()


def update_digest(h, batch_index, remote_ids):
    """Fold one batch. remote_ids: 1-D torch LongTensor or numpy int array
    (sampler order)."""
    arr = remote_ids.numpy() if hasattr(remote_ids, "numpy") else remote_ids
    h.update(int(batch_index).to_bytes(8, "little"))
    h.update(arr.astype("<i8").tobytes())


def digest_from_traces(paths, start_batch=0):
    """Recompute the rolling digest from dumped Trace npz files (epoch order).

    Returns (digest, n_batches). Equal to a live sampler_digest json iff the
    dump is a byte-exact record of the stream that run's sampler produced.
    """
    from optisched.trace import Trace
    h = new_digest()
    gb = start_batch
    for p in paths:
        tr = Trace.load(p)
        for b in range(tr.num_batches):
            nodes, _ = tr.batch(b)
            update_digest(h, gb, nodes)
            gb += 1
    return h, gb


def write_digest(path, h, n_batches, extra=None):
    rec = {"digest": h.hexdigest(), "n_batches": int(n_batches)}
    if extra:
        rec.update(extra)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


if __name__ == "__main__":
    # Verify a live trace dump against its digest record:
    #   python3 trace_digest.py <dump_dir> [digest_json]
    # digest_json defaults to the dump dir's own trace_part*_meta.json.
    import argparse
    import glob
    import os
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("digest_json", nargs="?", default=None)
    a = ap.parse_args()

    npz = sorted(glob.glob(os.path.join(a.dump_dir, "trace_part*_ep*.npz")))
    if not npz:
        sys.exit(f"no trace npz files in {a.dump_dir}")
    ref_path = a.digest_json or sorted(
        glob.glob(os.path.join(a.dump_dir, "trace_part*_meta.json")))[0]
    ref = json.load(open(ref_path))
    h, nb = digest_from_traces(npz)
    ok = (h.hexdigest() == ref["digest"]
          and nb == ref.get("n_batches", nb))
    print(f"recomputed={h.hexdigest()} n_batches={nb} "
          f"reference={ref['digest']} n_batches={ref.get('n_batches')} "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)
