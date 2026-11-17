#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge embed_*.py shard outputs back into one row-aligned (emb, meta) pair.

Why this exists
---------------
run_all.sh splits MoLFormer / ChemBERTa across the two GPUs with
``--shard i --num-shards 2``, so a full run leaves behind
``*_emb_shard0of2.npy`` + ``*_emb_shard1of2.npy``.  cluster_subset.py, however,
takes a *single* --emb and a *single* --meta and treats a comma-separated --emb
as "run the pipeline once per file", never as "concatenate".  Without this step
the full 3.45 M-molecule run cannot be clustered as one dataset.

Why it INTERLEAVES and does not concatenate
-------------------------------------------
Shard i holds ``df.iloc[i::num_shards]``, i.e. the *strided* slice
i, i+k, i+2k, ... .  Global row g therefore lives in shard ``g % k`` at local
position ``g // k``.  Concatenating shard0 then shard1 yields the order
0,2,4,...,1,3,5,... -- a permutation, not the original.  That is still
self-consistent (emb row j matches meta row j), so clustering one model alone
would look fine, but it silently breaks every cross-model comparison:
embed_morgan.py does not shard, so morgan_emb.npy stays in parquet order and
validate_emb.py would be comparing molecule g of one array against molecule
g' != g of another.  An earlier version of this script concatenated and its
docstring claimed otherwise; the assertion at the end of main() is what now
proves the order, instead of trusting a comment.

Order verification
------------------
After writing, the merged SMILES column is compared against the source parquet
(``data/smiles_all.parquet``, 3.45 M rows, ~1 s to read).  Any mismatch is a
hard error.  Use --no-verify-order only when the parquet cache is absent.

Memory safety
-------------
The full MoLFormer array is (3.45 M, 768) fp16 = 5.3 GB.  Both the source and
the destination are memory-mapped and the copy runs in chunks, so peak RSS stays
in the hundreds of MB regardless of dataset size.

Usage
-----
    python merge_shards.py --emb  'molformer/output/molformer_emb_shard*of2.npy' \
                           --meta 'molformer/output/molformer_smiles_shard*of2.csv'
    python merge_shards.py --emb  molformer/output/molformer_emb_shard0of2.npy \
                           --meta molformer/output/molformer_smiles_shard0of2.csv \
                           --out-emb  molformer/output/molformer_emb_merged.npy \
                           --out-meta molformer/output/molformer_smiles_merged.csv

Output paths default to the first input's directory with the ``shardXofY``
suffix replaced by ``merged``.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)                       # mol_io.py lives next to this script

SHARD_RE = re.compile(r"shard(\d+)of(\d+)")
COPY_CHUNK = 1 << 16     # rows gathered per pass; 64k rows ~ 95 MB at 768-d fp16
PARQUET = "/mnt/data/yangsensen/data/smiles_all.parquet"


def shard_index(path):
    """-> (shard, num_shards) parsed from the filename, or None."""
    m = SHARD_RE.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def resolve(patterns, kind):
    """Expand globs / comma lists -> {shard: path}, ordered by shard index.

    Returns (mapping, num_shards). num_shards == 1 means "one whole file, keep
    its order" (a non-sharded artefact such as morgan_emb.npy, or a shardXof1).
    """
    paths = []
    for pat in patterns.split(","):
        pat = pat.strip()
        if not pat:
            continue
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        if not hits:
            sys.exit("[merge] ERROR: no %s file matches %r" % (kind, pat))
        paths.extend(hits)

    mapping, counts = {}, {}
    for p in paths:
        si = shard_index(p)
        if si is None:
            # Not a sharded file: already whole, and there may be only one.
            if len(paths) != 1:
                sys.exit("[merge] ERROR: %s %r has no shardXofY tag but was passed "
                         "alongside other files; cannot infer an order" % (kind, p))
            return {0: p}, 1
        idx, ns = si          # shard_index() -> (shard, num_shards)
        if idx in mapping:
            sys.exit("[merge] ERROR: duplicate %s shard %d (%s and %s)"
                     % (kind, idx, mapping[idx], p))
        mapping[idx] = p
        counts.setdefault(ns, []).append(idx)

    if len(counts) > 1:
        sys.exit("[merge] ERROR: mixed --num-shards values %s" % sorted(counts))
    ns = next(iter(counts))
    got = counts[ns]
    missing = sorted(set(range(ns)) - set(got))
    if missing:
        sys.exit("[merge] ERROR: missing %s shard(s) %s of %d -- refusing to build "
                 "a silently truncated array" % (kind, missing, ns))
    print("[merge] %s: %d shard(s), indices %s" % (kind, ns, sorted(mapping)), flush=True)
    return mapping, ns


def default_out(first_path, suffix="_merged"):
    d, base = os.path.split(first_path)
    stem, ext = os.path.splitext(base)
    new_stem = SHARD_RE.sub("merged", stem)
    if new_stem == stem:
        new_stem += suffix
    return os.path.join(d, new_stem + ext)


def check_lengths(lens, ns, kind):
    """Shards of one strided split differ in length by at most one row."""
    if ns == 1:
        return int(lens[0])
    lo, hi = min(lens), max(lens)
    if hi - lo > 1:
        sys.exit("[merge] ERROR: %s shard lengths %s differ by more than one row. "
                 "Shards came from different runs or different --limit values; "
                 "re-run them together." % (kind, lens))
    total = int(sum(lens))
    print("[merge] %s: shard rows %s -> %d total" % (kind, lens, total), flush=True)
    return total


def merge_npy(mapping, ns, out_emb):
    """Interleave shard rows back into original parquet order, chunk by chunk."""
    srcs = {s: np.load(p, mmap_mode="r") for s, p in mapping.items()}
    dims = {v.shape[1:] for v in srcs.values()}
    if len(dims) != 1:
        sys.exit("[merge] ERROR: incompatible trailing dims %s" % sorted(map(str, dims)))
    dtypes = {v.dtype for v in srcs.values()}
    if len(dtypes) != 1:
        sys.exit("[merge] ERROR: mixed dtypes %s" % sorted(dtypes, key=str))

    total = check_lengths([len(v) for v in srcs.values()], ns, "emb")
    dtype = srcs[0].dtype
    print("[merge] writing %s  shape=(%d, %s) dtype=%s  (interleaved, %d shard%s)"
          % (out_emb, total, "x".join(map(str, srcs[0].shape[1:])), dtype, ns,
             "" if ns == 1 else "s"), flush=True)

    dst = np.lib.format.open_memmap(out_emb, mode="w+", dtype=dtype,
                                    shape=(total,) + srcs[0].shape[1:])
    for a in range(0, total, COPY_CHUNK):
        b = min(a + COPY_CHUNK, total)
        g = np.arange(a, b)
        if ns == 1:
            dst[a:b] = srcs[0][a:b]
            continue
        # Global row g came from shard g % ns, local position g // ns.
        for s in range(ns):
            sel = g % ns == s
            if not sel.any():
                continue
            dst[a:b][sel] = srcs[s][g[sel] // ns]
        if (a // COPY_CHUNK) % 8 == 0:
            print("[merge]   %d/%d rows (%.0f%%)" % (b, total, 100.0 * b / total),
                  flush=True)
    dst.flush()
    del dst
    return total


def merge_meta(mapping, ns, out_meta, expect_rows):
    """Apply the very same interleave to the SMILES tables."""
    frames = {s: pd.read_csv(p) for s, p in mapping.items()}
    cols = {tuple(f.columns) for f in frames.values()}
    if len(cols) != 1:
        sys.exit("[merge] ERROR: meta files have different columns: %s" % sorted(cols))
    if "smiles" not in frames[0].columns:
        sys.exit("[merge] ERROR: meta has no 'smiles' column (cluster_subset needs it)")

    total = check_lengths([len(f) for f in frames.values()], ns, "meta")
    if total != expect_rows:
        sys.exit("[merge] ERROR: row misalignment -- emb has %d rows but meta has %d. "
                 "Shards and their CSVs are out of sync; do not cluster this."
                 % (expect_rows, total))

    if ns == 1:
        meta = frames[0].reset_index(drop=True)
    else:
        # Position i of the merged table is global row i -> shard i % ns, row i // ns.
        loc = {s: np.arange(s, total, ns) // ns for s in range(ns)}
        meta = pd.concat([frames[s].iloc[loc[s]] for s in range(ns)])
        order = np.argsort(np.concatenate([np.arange(s, total, ns) for s in range(ns)]),
                           kind="stable")
        meta = meta.iloc[order].reset_index(drop=True)

    meta.to_csv(out_meta, index=False)
    print("[merge] wrote %s  rows=%d  cols=%s" % (out_meta, len(meta), list(meta.columns)),
          flush=True)
    return meta


def verify_contents(out_emb, meta):
    """Reopen the artefacts and confirm they agree with each other."""
    X = np.load(out_emb, mmap_mode="r")
    if len(X) != len(meta):
        sys.exit("[merge] ERROR: verify failed, %d emb rows vs %d meta rows"
                 % (len(X), len(meta)))
    head = np.asarray(X[:min(len(X), 4096)], dtype=np.float32)
    n_nan = int(np.isnan(head).sum())
    n_inf = int(np.isinf(head).sum())
    print("[merge] verify OK: reopened %s shape=%s dtype=%s | first %d rows NaN=%d Inf=%d"
          % (os.path.basename(out_emb), X.shape, X.dtype, len(head), n_nan, n_inf),
          flush=True)
    if n_nan or n_inf:
        sys.exit("[merge] ERROR: merged array contains NaN/Inf -- the shards are bad. "
                 "Run nan_check.py; for MoLFormer the compute dtype must be bfloat16.")


def verify_order(meta, parquet=PARQUET):
    """Prove the merged SMILES column really is the parquet's first N rows.

    This is the check that catches a concatenate-instead-of-interleave mistake.
    emb and meta always carry the same permutation, so comparing them to each
    other can never detect one; only the unsharded source can.
    """
    if not os.path.exists(parquet):
        print("[merge] WARN: %s not found, cannot prove row order -- SKIPPED" % parquet,
              flush=True)
        return False
    ref = pd.read_parquet(parquet, columns=["smiles"])["smiles"].values
    n = len(meta)
    if n > len(ref):
        sys.exit("[merge] ERROR: merged meta has %d rows but the parquet only has %d"
                 % (n, len(ref)))
    # Compared as Python objects on purpose: casting 3.45 M SMILES to numpy
    # unicode would allocate ~350 MB per side for no benefit.
    got = meta["smiles"].tolist()
    want = ref[:n].tolist()
    bad = int(sum(1 for a, b in zip(got, want) if a != b))
    if bad:
        first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        sys.exit("[merge] ERROR: merged row order does NOT match the source parquet "
                 "(%d/%d SMILES differ, first at row %d). Shards were concatenated "
                 "instead of interleaved, or came from a different --limit/--shard "
                 "configuration. Do not cluster this file." % (bad, n, first))
    print("[merge] order verified: all %d merged SMILES equal parquet rows 0..%d"
          % (n, n - 1), flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True,
                    help="shard .npy files: glob, comma list, or one plain path")
    ap.add_argument("--meta", required=True, help="matching smiles CSV(s), same syntax")
    ap.add_argument("--out-emb", default=None)
    ap.add_argument("--out-meta", default=None)
    ap.add_argument("--parquet", default=PARQUET, help="source SMILES table used to prove row order")
    ap.add_argument("--no-verify-order", action="store_true",
                    help="skip the parquet order check (only if the cache is missing)")
    args = ap.parse_args()

    emb_map, ns_e = resolve(args.emb, "emb")
    meta_map, ns_m = resolve(args.meta, "meta")
    if ns_e != ns_m or sorted(emb_map) != sorted(meta_map):
        sys.exit("[merge] ERROR: emb shards %s and meta shards %s do not describe the "
                 "same split" % (sorted(emb_map), sorted(meta_map)))

    out_emb = args.out_emb or default_out(emb_map[0])
    out_meta = args.out_meta or default_out(meta_map[0])
    if os.path.abspath(out_emb) in {os.path.abspath(p) for p in emb_map.values()}:
        sys.exit("[merge] ERROR: --out-emb would overwrite an input shard")

    n = merge_npy(emb_map, ns_e, out_emb)
    meta = merge_meta(meta_map, ns_e, out_meta, n)
    verify_contents(out_emb, meta)
    if ns_e > 1 and not args.no_verify_order:
        verify_order(meta, args.parquet)
    elif ns_e == 1:
        # Nothing was reordered (single file / shardXof1), so there is no
        # permutation to prove. Note morgan_emb.npy legitimately has fewer rows
        # than the parquet because RDKit drops unparseable SMILES.
        print("[merge] single input, no reordering -- parquet order check not applicable",
              flush=True)
    print("[merge] done\n  --emb  %s\n  --meta %s" % (out_emb, out_meta), flush=True)


if __name__ == "__main__":
    main()
