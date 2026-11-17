#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression test for merge_shards.py.

An earlier version concatenated shard0 + shard1 and documented that this
restores parquet order.  It does not: shard i is ``df.iloc[i::num_shards]``, so
concatenation yields 0,2,4,...,1,3,5,....  The permutation is invisible if you
only compare the merged emb against its own merged meta (both carry it), which is
how it survived review.  These cases rebuild shards exactly the way embed_*.py
does and compare the merge against the untouched source -- the only comparison
that can catch it.

Run:  python test_merge_shards.py          (local, no server needed)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE = os.path.join(HERE, "merge_shards.py")
DIM = 3


def smiles_of(i):
    return "S%04d" % i


def build_shards(tmp, n_rows, ns, limit=0):
    """Write emb/meta shards the way embed_*.py does."""
    full = pd.DataFrame({"smiles": [smiles_of(i) for i in range(n_rows)],
                         "split": ["train"] * n_rows})
    emb = np.arange(n_rows * DIM, dtype=np.float16).reshape(n_rows, DIM)
    for s in range(ns):
        d = full.iloc[s::ns].reset_index(drop=True)
        e = emb[s::ns]
        if limit:
            d, e = d.head(limit).reset_index(drop=True), e[:limit]
        tag = "shard%dof%d" % (s, ns)
        np.save(os.path.join(tmp, "x_emb_%s.npy" % tag), e)
        d.to_csv(os.path.join(tmp, "x_smiles_%s.csv" % tag), index=False)


def run_merge(tmp, ns):
    cmd = [sys.executable, MERGE,
           "--emb", os.path.join(tmp, "x_emb_shard*of%d.npy" % ns),
           "--meta", os.path.join(tmp, "x_smiles_shard*of%d.csv" % ns),
           "--out-emb", os.path.join(tmp, "x_emb_merged.npy"),
           "--out-meta", os.path.join(tmp, "x_smiles_merged.csv"),
           "--no-verify-order"]
    return subprocess.run(cmd, capture_output=True, text=True)


def check(name, cond, detail=""):
    print("  %-52s %s%s" % (name, "PASS" if cond else "*** FAIL ***",
                            "" if cond else "  <- " + detail))
    return bool(cond)


def case(n_rows, ns, limit=0):
    """Merging the shards must reproduce global rows 0..total-1, in order."""
    with tempfile.TemporaryDirectory() as tmp:
        build_shards(tmp, n_rows, ns, limit)
        total = min(n_rows, ns * limit) if limit else n_rows
        label = "N=%-4d ns=%d limit=%-3d -> %d rows" % (n_rows, ns, limit, total)

        r = run_merge(tmp, ns)
        if r.returncode != 0:
            return check(label, False,
                         "exit %d: %s" % (r.returncode, (r.stderr or r.stdout).strip()[-260:]))

        got_m = pd.read_csv(os.path.join(tmp, "x_smiles_merged.csv"))
        got_e = np.load(os.path.join(tmp, "x_emb_merged.npy"))
        exp_s = [smiles_of(i) for i in range(total)]
        exp_e = np.arange(total * DIM, dtype=np.float16).reshape(total, DIM)

        ok = True
        if got_m["smiles"].tolist() != exp_s:
            at = next(i for i, (a, b) in enumerate(zip(got_m["smiles"], exp_s)) if a != b)
            ok &= check(label + "  smiles order", False,
                        "row %d is %s, expected %s" % (at, got_m["smiles"][at], exp_s[at]))
        else:
            ok &= check(label + "  smiles order", True)
        ok &= check(label + "  emb order", np.array_equal(got_e, exp_e))
        # emb row j must encode molecule j: the vector starts at j*DIM.
        ok &= check(label + "  emb/meta same molecule",
                    all(int(v[0]) // DIM == j for j, v in enumerate(got_e)))
        return ok


def case_missing_shard():
    """A dropped shard must be a hard error, not a silently short array."""
    with tempfile.TemporaryDirectory() as tmp:
        build_shards(tmp, 40, 2)
        os.remove(os.path.join(tmp, "x_emb_shard1of2.npy"))
        r = run_merge(tmp, 2)
        said = (r.stderr or "") + (r.stdout or "")
        return check("missing shard -> hard error",
                     r.returncode != 0 and "missing" in said.lower(),
                     "exit %d: %s" % (r.returncode, said.strip()[-200:]))


def case_uneven_shards():
    """Shards from different --limit values must be rejected, not misaligned."""
    with tempfile.TemporaryDirectory() as tmp:
        build_shards(tmp, 40, 2)
        pd.DataFrame({"smiles": [smiles_of(i) for i in range(1, 40, 2)][:5],
                      "split": ["train"] * 5}).to_csv(
            os.path.join(tmp, "x_smiles_shard1of2.csv"), index=False)
        np.save(os.path.join(tmp, "x_emb_shard1of2.npy"),
                np.zeros((5, DIM), dtype=np.float16))
        r = run_merge(tmp, 2)
        return check("mismatched shard sizes -> hard error", r.returncode != 0,
                     "exit %d: %s" % (r.returncode, (r.stderr or r.stdout).strip()[-200:]))


def main():
    print("=" * 78)
    print("merge_shards.py regression test")
    print("=" * 78)
    ok = True
    print("-- full splits (no --limit)")
    for ns in (1, 2, 3, 4):
        for n_rows in (1, 2, 5, 7, 64, 101):
            ok &= case(n_rows, ns)
    print("-- with --limit (pilot path: union = first LIMIT*ns parquet rows)")
    for ns in (2, 3):
        for n_rows, limit in ((1000, 10), (1000, 100), (37, 5)):
            ok &= case(n_rows, ns, limit)
    print("-- failure modes")
    ok &= case_missing_shard()
    ok &= case_uneven_shards()
    print("=" * 78)
    print("RESULT: %s" % ("ALL PASS" if ok else "*** FAILURES ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
