# -*- coding: utf-8 -*-
"""Re-draw the subset from TRAIN rows only, reusing the existing full clustering.

What was wrong
--------------
The three 50k subsets under full_3.4m/{molformer,chemberta2,morgan}/ were drawn
from all 3,452,095 rows, valid included. PCQM4Mv2's valid split IS the evaluation
set, so 73,545 valid molecules were in the sampling pool and ~2.1% of every
subset (about 1,065 rows at 50k) came straight from it. Pretraining on those and
then scoring on valid is label leakage, and it inflates the number by an
uncontrolled amount.

What this does
--------------
The cluster labels are kept as they are -- `*_labels_all.npy` is the full-set
classification and does not need recomputing. We simply restrict the sampling
pool to split == "train" (3,378,550 rows, exactly the official PCQM4Mv2 train
set) and re-draw inside that pool. Clustering is a function of embeddings only
and never saw y, so reusing the labels costs nothing statistically; what matters
is that no valid row can be selected any more. This asserts it at the end.

Sampling rule (unchanged in spirit, now train-normalised)
---------------------------------------------------------
Per stratum (each cluster, plus the noise layer as its own stratum):

    quota_i = N_i_train * (target / N_train)      -- proportional, largest
                                                     remainder to hit target
                                                     exactly, floor of 1
    picks   = quota_i rows drawn uniformly at random without replacement

Proportional quota + uniform random within stratum is what makes the subset an
unbiased sample of the pool. sqrt-quota or medoid/FPS selection would flatten
big families and over-weight rare ones, i.e. deliberately depart from the full
distribution -- the opposite of the criterion here.

Nested prefixes
---------------
Each picked row also gets a `nest_rank` in 1..target, and the file is sorted by
it. The rank is built by sorting on the within-stratum draw position normalised
by that stratum's quota, so the first k rows of the file are themselves an
(approximately) proportional stratified sample of size k. That gives a subset
size sweep -- `head -n 50000` is a legitimate 50k subset -- from one draw,
without having to re-run anything to answer "was 140k the right size?".
Prefix TV is reported for each requested checkpoint.

Also emitted: a uniform-random subset of the same size from the same train pool
(the control arm). Stratification's value is only a real claim if it is measured
against that baseline with the same seed budget.

Usage
-----
    python resample_train_subset.py --target 140000 \
        --models molformer,chemberta2,morgan \
        --outdir full_3.4m/train_only_140k
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

DATA = "/mnt/data/yangsensen/data"
EMB = "/mnt/data/yangsensen/embedding"
LABEL_CACHE = os.path.join(DATA, "labels_all.parquet")
META_CSV = os.path.join(EMB, "molformer/output/molformer_smiles_merged.csv")
# labels_all.npy lives per model, with a per-model file stem.
LABEL_NPY = {
    "molformer": "full_3.4m/molformer/molformer_emb_merged_labels_all.npy",
    "chemberta2": "full_3.4m/chemberta2/chemberta2_emb_merged_labels_all.npy",
    "morgan": "full_3.4m/morgan/morgan_emb_labels_all.npy",
}


# --------------------------------------------------------------------------- #
# quota allocation
# --------------------------------------------------------------------------- #
def allocate(counts, target):
    """Proportional integer quotas summing exactly to `target`, each >= 1.

    Largest-remainder apportionment: floor first, then hand the leftover seats to
    the strata with the biggest fractional parts. Removing seats (only needed if
    the >=1 floor overshot) goes to the smallest fractional parts that can spare
    one, so no stratum is ever emptied -- that guarantee is the whole point of
    stratifying instead of drawing uniformly.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    exact = counts * (float(target) / float(n))
    q = np.maximum(np.floor(exact).astype(np.int64), 1)
    frac = exact - np.floor(exact)
    delta = int(target - q.sum())
    if delta > 0:
        order = np.argsort(-frac, kind="stable")
        i = 0
        while delta > 0:
            q[order[i % len(order)]] += 1
            delta -= 1
            i += 1
    elif delta < 0:
        cand = np.flatnonzero(q > 1)
        order = cand[np.argsort(frac[cand], kind="stable")]
        i = 0
        while delta < 0 and i < 100 * max(len(order), 1):
            j = order[i % len(order)]
            if q[j] > 1:
                q[j] -= 1
                delta += 1
            i += 1
        if delta != 0:
            raise SystemExit("[resample] could not reconcile quotas, off by %d" % delta)
    return q, exact


def stratified_pick(stratum_of_pool, pool_idx, target, seed):
    """-> (picked_pool_positions, nest_u) for one proportional stratified draw."""
    uniq, inv = np.unique(stratum_of_pool, return_inverse=True)
    inv = inv.ravel()
    counts = np.bincount(inv, minlength=len(uniq))
    q, _ = allocate(counts, target)

    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))

    picks, us = [], []
    for i in range(len(uniq)):
        members = order[bounds[i]:bounds[i + 1]]
        k = int(q[i])
        # Per-stratum seed derived from (seed, stratum number) rather than one
        # shared stream: the draw then does not depend on iteration order, and
        # re-running a single stratum reproduces the same rows.
        r = np.random.default_rng([seed, int(uniq[i]) + 1])
        take = r.choice(len(members), size=k, replace=False)
        picks.append(members[take])
        us.append((np.arange(k, dtype=np.float64) + 0.5) / k)

    picks = np.concatenate(picks)
    us = np.concatenate(us)
    return pool_idx[picks], us, uniq, counts, q


def nest_rank(us, seed):
    """Global 1..n ordering whose every prefix is proportionally stratified."""
    tie = np.random.default_rng([seed, 4242]).random(len(us))
    key = us + 1e-9 * tie
    return np.argsort(key, kind="stable") + 1


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
def hist_tv(a, b, lo, hi, bins=200):
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = ha / max(ha.sum(), 1)
    pb = hb / max(hb.sum(), 1)
    return float(0.5 * np.abs(pa - pb).sum())


def y_block(y_sub, y_pool, seed, n_null=20):
    """y-distribution agreement, judged against a random-subset null.

    With n = 140k a KS p-value against a 3.38 M reference is ~0 for ANY sample,
    including a perfectly uniform random one, so a bare p-value would be
    meaningless and a reviewer would say so. The honest form is: how does this
    subset's KS compare with the KS of same-size uniform random draws? If it sits
    inside that null band, the subset is indistinguishable from random on y.
    """
    ks = stats.ks_2samp(y_sub, y_pool)
    nulls = []
    r = np.random.default_rng([seed, 31337])
    for _ in range(n_null):
        idx = r.choice(len(y_pool), size=len(y_sub), replace=False)
        nulls.append(float(stats.ks_2samp(y_pool[idx], y_pool).statistic))
    nulls = np.asarray(nulls)
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    lo, hi = float(np.min(y_pool)), float(np.max(y_pool))
    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "ks_null_random_mean": float(nulls.mean()),
        "ks_null_random_max": float(nulls.max()),
        "ks_pct_of_null": float((nulls < ks.statistic).mean()),
        "hist_tv_200bin": hist_tv(y_sub, y_pool, lo, hi),
        "mean": float(np.mean(y_sub)),
        "pool_mean": float(np.mean(y_pool)),
        "std": float(np.std(y_sub)),
        "pool_std": float(np.std(y_pool)),
        "skew": float(stats.skew(y_sub)),
        "pool_skew": float(stats.skew(y_pool)),
        "kurtosis": float(stats.kurtosis(y_sub)),
        "pool_kurtosis": float(stats.kurtosis(y_pool)),
        "quantiles": {("%g" % q): [float(np.quantile(y_sub, q)), float(np.quantile(y_pool, q))]
                      for q in qs},
        "n_null_draws": n_null,
    }


def strata_block(stratum_of_pool, counts, q, picked_stratum, target):
    """Per-stratum proportionality: rates, TV, chi-square, binomial z, miss risk."""
    n_pool = int(counts.sum())
    p = target / n_pool
    rate = q / np.maximum(counts, 1)
    tv = float(0.5 * np.abs(q / target - counts / n_pool).sum())

    exp = counts * p
    chi2 = float(((q - exp) ** 2 / np.maximum(exp, 1e-12)).sum())
    df = max(len(counts) - 1, 1)
    chi2_p = float(stats.chi2.sf(chi2, df))

    sd = np.sqrt(np.maximum(exp * (1.0 - p), 1e-12))
    z = (q - exp) / sd

    # A uniform random draw of the same size omits a stratum of size N with
    # probability (1-p)^N. Summing over strata gives the expected number of
    # chemical families a random subset would miss entirely -- the quantity that
    # stratification drives to exactly zero.
    miss = float(np.sum(np.exp(counts * np.log1p(-p))))

    table = pd.DataFrame({
        "stratum": picked_stratum,
        "n_train": counts,
        "quota": q,
        "rate": rate,
        "expected": exp,
        "z": z,
        "p_missed_by_random": np.exp(counts * np.log1p(-p)),
    })
    return {
        "n_strata": int(len(counts)),
        "n_pool": n_pool,
        "target": int(target),
        "global_rate": p,
        "rate_min": float(rate.min()),
        "rate_median": float(np.median(rate)),
        "rate_max": float(rate.max()),
        "strata_with_zero_picks": int((q == 0).sum()),
        "mass_tv": tv,
        "chi2": chi2,
        "chi2_df": df,
        "chi2_pvalue": chi2_p,
        "binom_z_frac_within_1.96": float(np.mean(np.abs(z) < 1.96)),
        "binom_z_mean": float(z.mean()),
        "binom_z_std": float(z.std()),
        "expected_strata_missed_by_random": miss,
    }, table


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=140000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="molformer,chemberta2,morgan")
    ap.add_argument("--label-cache", default=LABEL_CACHE)
    ap.add_argument("--meta-csv", default=META_CSV,
                    help="the smiles CSV the embedders consumed; used only to "
                         "verify index alignment with the label cache")
    ap.add_argument("--emb-root", default=EMB)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--prefixes", default="20000,50000,100000",
                    help="comma separated nested-prefix sizes to QA")
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--no-random-baseline", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    outdir = args.outdir or os.path.join(args.emb_root, "full_3.4m",
                                         "train_only_%dk" % (args.target // 1000))
    os.makedirs(outdir, exist_ok=True)
    prefixes = sorted({int(x) for x in args.prefixes.split(",") if x.strip()} | {args.target})
    t_start = time.time()

    lab = pd.read_parquet(args.label_cache)
    n_all = len(lab)
    print("[resample] label cache %s rows=%d cols=%s" % (args.label_cache, n_all, list(lab.columns)),
          flush=True)

    # The embeddings, UMAP coords and cluster labels were all written in the
    # order of the meta CSV the embedders read. If that order is not the order of
    # the label cache, every y attached below would belong to a different
    # molecule -- silently. So check it once, on all rows, and abort otherwise.
    if args.meta_csv and os.path.exists(args.meta_csv):
        tm = time.time()
        meta = pd.read_csv(args.meta_csv, usecols=["smiles", "split"], dtype=str)
        bad_smi = int((meta["smiles"].to_numpy() != lab["smiles"].astype("object").to_numpy()).sum())
        bad_sp = int((meta["split"].to_numpy() != lab["split"].astype("object").to_numpy()).sum())
        print("[resample] alignment vs %s: rows=%d smiles mismatches=%d split mismatches=%d (%.0fs)"
              % (os.path.basename(args.meta_csv), len(meta), bad_smi, bad_sp, time.time() - tm),
              flush=True)
        if len(meta) != n_all or bad_smi or bad_sp:
            raise SystemExit("[resample] FATAL: meta CSV and label cache disagree; "
                             "refusing to attach y to cluster labels.")
    else:
        print("[resample] WARNING: meta CSV %s absent, alignment NOT verified" % args.meta_csv,
              flush=True)

    split = lab["split"].astype("object").to_numpy()
    smiles = lab["smiles"].astype("object").to_numpy()
    y_all = lab["y"].to_numpy(dtype=np.float64)
    if not np.isfinite(y_all).all():
        raise SystemExit("[resample] FATAL: %d non-finite y values in the cache"
                         % int((~np.isfinite(y_all)).sum()))

    train_idx = np.flatnonzero(split == "train")
    n_train = len(train_idx)
    n_valid = int((split == "valid").sum())
    y_pool = y_all[train_idx]
    print("[resample] pool: train=%d valid=%d (valid excluded) target=%d -> rate=%.6f%%"
          % (n_train, n_valid, args.target, 100.0 * args.target / n_train), flush=True)
    if args.target > n_train:
        raise SystemExit("[resample] target %d exceeds train pool %d" % (args.target, n_train))

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label_cache": args.label_cache,
        "meta_csv_verified": bool(args.meta_csv and os.path.exists(args.meta_csv)),
        "n_all": int(n_all),
        "n_train": int(n_train),
        "n_valid_excluded": int(n_valid),
        "target": int(args.target),
        "seed": int(args.seed),
        "sampling": "proportional quota per stratum (largest remainder, min 1) + "
                    "uniform random within stratum; noise layer is its own stratum",
        "strata_source": "existing full-set *_labels_all.npy (reused, not recomputed)",
        "prefixes": prefixes,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "models": {},
    }

    for name in models:
        tm = time.time()
        rel = LABEL_NPY.get(name)
        if rel is None:
            print("[resample] !! unknown model %r, skipping" % name, flush=True)
            continue
        npy = os.path.join(args.emb_root, rel)
        L = np.load(npy)
        print("\n=== %s: %s shape=%s dtype=%s" % (name, os.path.basename(npy), L.shape, L.dtype),
              flush=True)
        if len(L) != n_all:
            raise SystemExit("[resample] FATAL %s: labels have %d rows, cache has %d"
                             % (name, len(L), n_all))

        lab_pool = L[train_idx]
        picked, us, uniq, counts, q = stratified_pick(lab_pool, train_idx, args.target, args.seed)
        ranks = nest_rank(us, args.seed)

        sub = pd.DataFrame({
            "nest_rank": ranks,
            "idx": picked,
            "smiles": smiles[picked],
            "y": y_all[picked],
            "split": split[picked],
            "cluster": L[picked],
        })
        sub["role"] = np.where(sub["cluster"].to_numpy() < 0, "noise", "cluster")
        sub = sub.sort_values("nest_rank", kind="stable").reset_index(drop=True)

        n_valid_in = int((sub["split"] != "train").sum())
        if n_valid_in:
            raise SystemExit("[resample] FATAL %s: %d non-train rows in the subset"
                             % (name, n_valid_in))

        strata_picked = sub["cluster"].to_numpy()
        sb, table = strata_block(uniq, counts, q, uniq, args.target)
        yb = y_block(sub["y"].to_numpy(dtype=np.float64), y_pool, args.seed, args.n_null)

        # nested prefixes: the first k rows must themselves be proportional
        pref = {}
        for k in prefixes:
            if k > len(sub):
                continue
            sk = strata_picked[:k]
            cnt = np.bincount(np.searchsorted(uniq, sk), minlength=len(uniq)).astype(np.float64)
            pref[str(k)] = {
                "mass_tv": float(0.5 * np.abs(cnt / k - counts / counts.sum()).sum()),
                "strata_covered": int((cnt > 0).sum()),
                "ks_y": float(stats.ks_2samp(sub["y"].to_numpy(dtype=np.float64)[:k], y_pool).statistic),
            }

        rec = {
            "labels_npy": rel,
            "n_strata_full": int(len(np.unique(L))),
            "subset_csv": os.path.basename(os.path.join(outdir, "%s_train_subset_%dk.csv" % (name, args.target // 1000))),
            "strata": sb,
            "y": yb,
            "nested_prefixes": pref,
            "n_rows": int(len(sub)),
            "n_valid_rows": n_valid_in,
            "n_unique_smiles": int(sub["smiles"].nunique()),
            "noise_rows": int((sub["role"] == "noise").sum()),
            "noise_share_subset": float((sub["role"] == "noise").mean()),
            "noise_share_pool": float((lab_pool < 0).mean()),
            "seconds": round(time.time() - tm, 1),
        }
        print("[resample] %s: picked=%d strata=%d rate med=%.6f%% [%.4f%%,%.4f%%] TV=%.5f "
              "chi2 p=%.3f miss_random=%.2f KS_y=%.5f (null %.5f) noise %.2f%% vs pool %.2f%%"
              % (name, len(sub), sb["n_strata"], 100 * sb["rate_median"],
                 100 * sb["rate_min"], 100 * sb["rate_max"], sb["mass_tv"], sb["chi2_pvalue"],
                 sb["expected_strata_missed_by_random"], yb["ks_stat"],
                 yb["ks_null_random_mean"], 100 * rec["noise_share_subset"],
                 100 * rec["noise_share_pool"]), flush=True)

        # ---- uniform random control arm, same pool, same size ----
        if not args.no_random_baseline:
            r = np.random.default_rng([args.seed, 313337])
            rp = train_idx[r.choice(n_train, size=args.target, replace=False)]
            rsub = pd.DataFrame({
                "nest_rank": np.arange(1, args.target + 1),
                "idx": rp,
                "smiles": smiles[rp],
                "y": y_all[rp],
                "split": split[rp],
                "cluster": L[rp],
            })
            rsub["role"] = np.where(rsub["cluster"].to_numpy() < 0, "noise", "cluster")
            rcnt = np.bincount(np.searchsorted(uniq, rsub["cluster"].to_numpy()),
                               minlength=len(uniq)).astype(np.int64)
            rks = float(stats.ks_2samp(rsub["y"].to_numpy(dtype=np.float64), y_pool).statistic)
            rtv = float(0.5 * np.abs(rcnt / args.target - counts / counts.sum()).sum())
            rec["random_baseline"] = {
                "csv": "%s_train_random_%dk.csv" % (name, args.target // 1000),
                "n_rows": int(len(rsub)),
                "n_valid_rows": int((rsub["split"] != "train").sum()),
                "n_unique_smiles": int(rsub["smiles"].nunique()),
                "mass_tv": rtv,
                "ks_y": rks,
                "strata_with_zero_picks": int((rcnt == 0).sum()),
                "expected_strata_missed_by_random": sb["expected_strata_missed_by_random"],
                "noise_share_subset": float((rsub["role"] == "noise").mean()),
            }
            print("[resample] %s: random arm TV=%.5f KS_y=%.5f empty strata=%d"
                  % (name, rtv, rks, int((rcnt == 0).sum())), flush=True)
            rpath = os.path.join(outdir, rec["random_baseline"]["csv"])
            rsub.to_csv(rpath, index=False)

        spath = os.path.join(outdir, rec["subset_csv"])
        sub.to_csv(spath, index=False)
        tpath = os.path.join(outdir, "%s_cluster_table_%dk.csv" % (name, args.target // 1000))
        table.sort_values("n_train", ascending=False).to_csv(tpath, index=False)
        rec["cluster_table_csv"] = os.path.basename(tpath)
        summary["models"][name] = rec
        print("[resample] %s: wrote %s (%.1fs)" % (name, spath, time.time() - tm), flush=True)

    jpath = os.path.join(outdir, "resample_qa_summary.json")
    with open(jpath, "w") as fh:
        json.dump(summary, fh, indent=2)
    cfg = os.path.join(outdir, "resample_run_config.json")
    with open(cfg, "w") as fh:
        json.dump({k: v for k, v in vars(args).items()}, fh, indent=2, default=str)
    print("\n[resample] wrote %s" % jpath, flush=True)
    print("[resample] DONE in %.0fs" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
