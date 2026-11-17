# -*- coding: utf-8 -*-
"""Parse PCQM4Mv2 *.jsonl.gz once into a parquet cache.

Why: every embedder used to re-parse the 1.1 GB of gzipped JSONL (3.37 M lines,
several minutes each, and 4x in total for 2 shards x 2 models). This builds
/mnt/data/yangsensen/data/smiles_all.parquet once; the embedders read it if it
exists and fall back to parsing the gz files otherwise.

Usage: python build_smiles_cache.py [--force]
"""
import argparse
import glob
import gzip
import json
import os
import time

import pandas as pd

DATA = "/mnt/data/yangsensen/data"
CACHE = os.path.join(DATA, "smiles_all.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--out", default=CACHE)
    ap.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        df = pd.read_parquet(args.out)
        print("cache already exists:", args.out, df.shape, flush=True)
        print(df["split"].value_counts().to_dict(), flush=True)
        return

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl.gz")))
    if not files:
        raise SystemExit("no *.jsonl.gz found under %s" % args.data_dir)
    print("parsing %d files" % len(files), flush=True)

    rows = []
    t0 = time.time()
    for f in files:
        split = "valid" if "valid" in os.path.basename(f) else "train"
        n0 = len(rows)
        with gzip.open(f, "rt") as fh:
            for line in fh:
                try:
                    smi = json.loads(line)["labels"]["object"]["SMILES"]
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append((smi, split))
        print("  %s: +%d rows (total %d, %.0fs)"
              % (os.path.basename(f), len(rows) - n0, len(rows), time.time() - t0), flush=True)

    df = pd.DataFrame(rows, columns=["smiles", "split"])
    tmp = "%s.tmp%d" % (args.out, os.getpid())
    df.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)          # atomic, never leaves a half-written cache
    print("saved %s  shape=%s  in %.0fs" % (args.out, df.shape, time.time() - t0), flush=True)
    print("splits:", df["split"].value_counts().to_dict(), flush=True)
    print("unique SMILES:", df["smiles"].nunique(), flush=True)


if __name__ == "__main__":
    main()
