# -*- coding: utf-8 -*-
"""Parse PCQM4Mv2 *.jsonl.gz into a label cache that carries the target y.

Why this exists
---------------
`smiles_all.parquet` has only (smiles, split). Every check that matters for the
downstream experiment needs y = labels.object.gap (HOMO-LUMO gap, eV):
the y-distribution KS/TV test, ANOVA eta^2, the proxy regression, and -- most
practically -- the subset CSV itself, so that whoever trains on the subset never
has to re-join 140k rows back against 1.1 GB of gzipped JSONL.

Index alignment is verified, not assumed
----------------------------------------
Row order here is byte-for-byte the recipe of build_smiles_cache.py:
sorted(glob(*.jsonl.gz)), one row per line, lines without labels.object.SMILES
skipped. After building, this script compares its own smiles/split columns to
smiles_all.parquet row-for-row and aborts on any mismatch. That comparison is
what licenses reusing the existing *_labels_all.npy / *_umap_all.npy arrays:
those were produced in this same index space, so once the smiles column agrees
with the cache the embedders consumed, cluster label i refers to row i here.

Why a regex instead of json.loads
---------------------------------
Each record carries a per-atom `position` array plus node_class/unitcell, so a
full parse of 3.45 M records spends nearly all its time on fields we discard.
The SMILES regex below is escape-aware (\\\\ and \\" cannot terminate a match),
and the captured token is passed through json.loads, so JSON escaping is decoded
by the real decoder rather than by hand. Combined with the row-for-row check
against the existing cache, the fast path cannot silently drift.

Usage
-----
    python build_label_cache.py [--force]
"""
import argparse
import glob
import gzip
import json
import os
import re
import time

import numpy as np
import pandas as pd

DATA = "/mnt/data/yangsensen/data"
SMILES_CACHE = os.path.join(DATA, "smiles_all.parquet")
LABEL_CACHE = os.path.join(DATA, "labels_all.parquet")

# "SMILES": "..." -- the value may contain escaped backslashes/quotes, which are
# common in stereo SMILES (e.g. /C(O)=N\\C(=N)O arrives as /C(O)=N\\\\C(=N)O).
SMI_RE = re.compile(rb'"SMILES"\s*:\s*"((?:[^"\\]|\\.)*)"')
GAP_RE = re.compile(rb'"gap"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default=LABEL_CACHE)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        df = pd.read_parquet(args.out)
        print("[labels] cache exists: %s shape=%s" % (args.out, df.shape), flush=True)
        print("[labels] splits: %s" % df["split"].value_counts().to_dict(), flush=True)
        print("[labels] y NaN: %d" % int(df["y"].isna().sum()), flush=True)
        return

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl.gz")))
    if not files:
        raise SystemExit("[labels] no *.jsonl.gz under %s" % args.data_dir)
    print("[labels] parsing %d files" % len(files), flush=True)

    smiles, split, y = [], [], []
    n_bad_smi = n_bad_gap = 0
    t0 = time.time()
    for f in files:
        sp = "valid" if "valid" in os.path.basename(f) else "train"
        n0 = len(smiles)
        with gzip.open(f, "rb") as fh:
            for line in fh:
                m = SMI_RE.search(line)
                if m is None:
                    # Same skip rule as build_smiles_cache.py: a line without a
                    # usable SMILES never entered the cache, so it must not
                    # enter here either or every later index shifts by one.
                    n_bad_smi += 1
                    continue
                try:
                    smi = json.loads('"' + m.group(1).decode("utf-8") + '"')
                except (UnicodeDecodeError, ValueError):
                    n_bad_smi += 1
                    continue
                g = GAP_RE.search(line)
                if g is None:
                    n_bad_gap += 1
                    val = np.nan
                else:
                    val = float(g.group(1))
                smiles.append(smi)
                split.append(sp)
                y.append(val)
        print("[labels]   %s: +%d rows (total %d, %.0fs)"
              % (os.path.basename(f), len(smiles) - n0, len(smiles), time.time() - t0),
              flush=True)

    df = pd.DataFrame({
        "idx": np.arange(len(smiles), dtype=np.int32),
        "smiles": pd.Series(smiles, dtype="string"),
        "split": pd.Series(split, dtype="string"),
        "y": pd.Series(np.asarray(y, dtype=np.float64)),
    })
    print("[labels] parsed rows=%d skipped_no_smiles=%d missing_gap=%d in %.0fs"
          % (len(df), n_bad_smi, n_bad_gap, time.time() - t0), flush=True)

    # ---- integrity check against the cache the embedders actually consumed ----
    if os.path.exists(SMILES_CACHE):
        ref = pd.read_parquet(SMILES_CACHE)
        if len(ref) != len(df):
            raise SystemExit("[labels] FATAL row count mismatch: %s has %d rows, "
                             "parsed %d -- index space would be shifted"
                             % (SMILES_CACHE, len(ref), len(df)))
        bad_smi = int((ref["smiles"].to_numpy() != df["smiles"].astype("object").to_numpy()).sum())
        bad_sp = int((ref["split"].to_numpy() != df["split"].astype("object").to_numpy()).sum())
        print("[labels] verify vs %s: smiles mismatches=%d split mismatches=%d"
              % (os.path.basename(SMILES_CACHE), bad_smi, bad_sp), flush=True)
        if bad_smi or bad_sp:
            raise SystemExit("[labels] FATAL: parsed order does not match the cache "
                             "the embeddings were built from; refusing to write.")
    else:
        print("[labels] WARNING: %s absent, index alignment NOT verified" % SMILES_CACHE,
              flush=True)

    tmp = "%s.tmp%d" % (args.out, os.getpid())
    df.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    print("[labels] saved %s shape=%s" % (args.out, df.shape), flush=True)
    for sp in ("train", "valid"):
        s = df.loc[df["split"] == sp, "y"]
        if len(s):
            print("[labels]   %-5s n=%d y mean=%.6f std=%.6f min=%.6f max=%.6f nan=%d"
                  % (sp, len(s), s.mean(), s.std(), s.min(), s.max(), int(s.isna().sum())),
                  flush=True)


if __name__ == "__main__":
    main()
