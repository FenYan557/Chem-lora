# -*- coding: utf-8 -*-
"""Bemis-Murcko scaffold distribution: subset vs the full TRAIN pool.

Why this file exists separately
-------------------------------
Everything in resample_train_subset.py measures agreement in spaces derived from
the embeddings (cluster strata) or in the target y. Both are internal to the
pipeline. Scaffold frequency is the one check that uses neither: it comes
straight from RDKit's perception of the ring systems, so it can catch a
distributional shift the embeddings happen to be blind to. It is also the number
a medicinal-chemistry reviewer looks at first -- "does this subset contain the
same chemical families, in the same proportions, as the dataset?"

Scope note: this compares distributions (claim A, "the subset is a fair
sample"). It deliberately does not run the per-cluster scaffold-concentration
permutation test (claim B, "clusters are chemically real"); that reuses the
cache built here, so it is a cheap follow-up rather than a rerun.

Cost: scaffold perception over the 3.32 M unique SMILES, parallel over 32 cores,
then cached to data/scaffold_cache.parquet so all models share one computation.

Usage
-----
    python scaffold_qa.py --subset-dir full_3.4m/train_only_140k
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

DATA = "/mnt/data/yangsensen/data"
CACHE = os.path.join(DATA, "scaffold_cache.parquet")
LABEL_CACHE = os.path.join(DATA, "labels_all.parquet")


def _scaf(smiles):
    """Generic Bemis-Murcko scaffold SMILES; '' for acyclic molecules.

    includeChirality=False: the generic scaffold keeps ring systems and linkers
    but strips atom stereo and side chains, which is the granularity at which
    "chemical family" is normally meant.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                out.append("__PARSE_FAIL__")
                continue
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            out.append(sc if sc else "__ACYCLIC__")
        except Exception:
            out.append("__PARSE_FAIL__")
    return out


def build_cache(uniq_smiles, procs, chunk=20000):
    t0 = time.time()
    res = []
    with Pool(procs) as pool:
        for i, part in enumerate(pool.imap(_scaf, (uniq_smiles[k:k + chunk]
                                                  for k in range(0, len(uniq_smiles), chunk)))):
            res.extend(part)
            if i % 20 == 0:
                print("[scaffold]   %d / %d (%.0fs)" % (len(res), len(uniq_smiles),
                                                        time.time() - t0), flush=True)
    return np.asarray(res, dtype=object)


def dist_metrics(sub_scaf, pool_scaf, top_n=50):
    """Frequency agreement between subset and pool over the scaffold vocabulary."""
    pc = pd.Series(pool_scaf).value_counts()
    sc = pd.Series(sub_scaf).value_counts()
    vocab = pc.index
    n_pool = int(pc.sum())
    n_sub = int(sc.sum())

    # TV over the pool vocabulary; scaffolds absent from the pool contribute
    # their whole subset mass (should be exactly zero -- subset is a subset).
    aligned = sc.reindex(vocab, fill_value=0).to_numpy(dtype=np.float64)
    extra = int(sc.index.difference(vocab).size)
    extra_mass = float(sc.loc[sc.index.difference(vocab)].sum()) if extra else 0.0
    tv = float(0.5 * (np.abs(aligned / n_sub - pc.to_numpy(dtype=np.float64) / n_pool).sum()
                      + extra_mass / n_sub))

    top = pc.head(top_n)
    top_sub = sc.reindex(top.index, fill_value=0).to_numpy(dtype=np.float64) / n_sub
    top_pool = top.to_numpy(dtype=np.float64) / n_pool

    # Frequency-weighted coverage: what share of pool molecules belong to a
    # scaffold that the subset actually contains. Distinct-count coverage alone
    # is misleading here, because most scaffolds in PCQM4Mv2 are singletons and a
    # 4% sample cannot contain them all -- by construction, not by bias.
    covered = pc.index.isin(sc.index)
    return {
        "n_scaffolds_pool": int(len(pc)),
        "n_scaffolds_subset": int(len(sc)),
        "distinct_coverage": float(len(sc) / len(pc)),
        "freq_weighted_coverage": float(pc.to_numpy()[covered].sum() / n_pool),
        "scaffolds_not_in_pool": extra,
        "mass_not_in_pool": extra_mass,
        "tv_all_scaffolds": tv,
        "top%d_share_subset" % top_n: float(top_sub.sum()),
        "top%d_share_pool" % top_n: float(top_pool.sum()),
        "top%d_max_abs_share_gap" % top_n: float(np.abs(top_sub - top_pool).max()),
        "top%d_table" % top_n: [
            {"scaffold": str(k), "pool_n": int(pc.loc[k]),
             "pool_share": float(pc.loc[k]) / n_pool,
             "subset_n": int(sc.get(k, 0)),
             "subset_share": float(sc.get(k, 0)) / n_sub}
            for k in top.index
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-dir", required=True)
    ap.add_argument("--pattern", default="*_train_*_*.csv",
                    help="glob inside subset-dir; matches both the stratified "
                         "subsets and the random control arms")
    ap.add_argument("--label-cache", default=LABEL_CACHE)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    lab = pd.read_parquet(args.label_cache)
    train = lab["split"].astype("object").to_numpy() == "train"
    train_smiles = lab.loc[train, "smiles"].astype("object").to_numpy()
    print("[scaffold] train rows=%d" % len(train_smiles), flush=True)

    uniq = pd.Index(pd.unique(train_smiles))
    print("[scaffold] unique train SMILES=%d" % len(uniq), flush=True)

    if os.path.exists(args.cache):
        c = pd.read_parquet(args.cache)
        if len(c) == len(uniq) and (c["smiles"].astype("object").to_numpy() == uniq.to_numpy()).all():
            scaf = c["scaffold"].astype("object").to_numpy()
            print("[scaffold] loaded cache %s" % args.cache, flush=True)
        else:
            print("[scaffold] cache stale, recomputing", flush=True)
            scaf = build_cache(list(uniq), args.procs)
    else:
        scaf = build_cache(list(uniq), args.procs)
        pd.DataFrame({"smiles": uniq.to_numpy(), "scaffold": scaf}).to_parquet(args.cache, index=False)
        print("[scaffold] saved cache %s" % args.cache, flush=True)

    lut = pd.DataFrame({"smiles": uniq.to_numpy(), "scaffold": scaf})
    pool_scaf = pd.DataFrame({"smiles": train_smiles}).merge(lut, on="smiles", how="left")["scaffold"]
    if int(pool_scaf.isna().sum()):
        raise SystemExit("[scaffold] FATAL: %d train rows failed to map to a scaffold"
                         % int(pool_scaf.isna().sum()))
    pool_scaf = pool_scaf.to_numpy()
    print("[scaffold] pool scaffolds: distinct=%d acyclic=%d parse_fail=%d"
          % (len(pd.unique(pool_scaf)), int((pool_scaf == "__ACYCLIC__").sum()),
             int((pool_scaf == "__PARSE_FAIL__").sum())), flush=True)

    files = sorted(glob.glob(os.path.join(args.subset_dir, args.pattern)))
    if not files:
        raise SystemExit("[scaffold] no subset CSVs matching %s in %s"
                         % (args.pattern, args.subset_dir))
    out = {"train_rows": int(len(train_smiles)),
           "unique_train_smiles": int(len(uniq)),
           "subsets": {}}
    for f in files:
        tm = time.time()
        df = pd.read_csv(f, usecols=["smiles"])
        s = df.merge(lut, on="smiles", how="left")["scaffold"]
        bad = int(s.isna().sum())
        if bad:
            raise SystemExit("[scaffold] FATAL %s: %d subset SMILES not found in the "
                             "train scaffold table -- the subset would not be a subset"
                             % (os.path.basename(f), bad))
        m = dist_metrics(s.to_numpy(), pool_scaf, args.top_n)
        m["csv"] = os.path.basename(f)
        m["n_rows"] = int(len(df))
        out["subsets"][os.path.basename(f)] = m
        print("[scaffold] %-42s TV=%.5f freqcov=%.4f distinct=%.3f top%d gap=%.5f (%.0fs)"
              % (m["csv"], m["tv_all_scaffolds"], m["freq_weighted_coverage"],
                 m["distinct_coverage"], args.top_n, m["top%d_max_abs_share_gap" % args.top_n],
                 time.time() - tm), flush=True)

    jp = os.path.join(args.subset_dir, "scaffold_qa.json")
    with open(jp, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[scaffold] wrote %s" % jp, flush=True)

    # flat CSV of the top-N table for every subset, easier to paste into a paper
    rows = []
    for name, m in out["subsets"].items():
        for r in m["top%d_table" % args.top_n]:
            rows.append({"subset": name, **r})
    cp = os.path.join(args.subset_dir, "scaffold_top%d_shares.csv" % args.top_n)
    pd.DataFrame(rows).to_csv(cp, index=False)
    print("[scaffold] wrote %s" % cp, flush=True)


if __name__ == "__main__":
    main()
