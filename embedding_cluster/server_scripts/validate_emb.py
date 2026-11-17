#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integrity + semantic validation of the embeddings before trusting a subset.

The 50 k run came out 4-25x faster than the handoff doc predicted (~6.7 k mol/s
for MoLFormer vs a recorded 1.65 k, ~22 k mol/s for ChemBERTa vs 860).  Speed
alone cannot tell us whether that is "the smoke test was warmup-bound" or "the
forward pass is silently degenerate", so this script checks the vectors
themselves:

  1. integrity  - shape / dtype / NaN / Inf / all-zero rows
  2. collapse   - per-dim variance and the pairwise-cosine distribution; a
                  collapsed encoder shows up as cosine ~1 everywhere
  3. semantics  - Spearman agreement with Morgan/Tanimoto, a learning-free
                  structural reference.  A real chemical encoder must rank
                  molecular similarity broadly the same way; ~0 means the
                  vectors carry no structural signal.  Reported twice: over all
                  pairs, and over the pairs where Tanimoto > 0.  The second
                  number is the one to read -- with ~29 on-bits out of 2048,
                  roughly two thirds of random pairs share no substructure at
                  all, so Tanimoto has one enormous tie block at 0 and Spearman
                  over all pairs is heavily attenuated by it.  A tie-blind rho
                  of 0.3 therefore does NOT mean "the encoder is 30% as good as
                  a fingerprint".
  4. neighbours - top-k nearest neighbours for a few queries, printed as SMILES
                  so the agreement can be eyeballed, not just trusted as a rho
  5. throughput - token-length stats + the implied TFLOP/s, checked against the
                  RTX 3090 fp16 ceiling.  If the measured mol/s needs more FLOPs
                  than the card can deliver, the timing is wrong, not the model.

Row alignment used here: merge_shards.py restores original parquet order, so
merged_molformer[j] and merged_chemberta[j] and morgan[j] are all global row j.
Every comparison asserts the SMILES strings match before trusting the vectors.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

EMB = "/mnt/data/yangsensen/embedding"
RNG = np.random.default_rng(0)

# RTX 3090 (sm_86): 35.6 TFLOP/s fp16 tensor-core with fp32 accumulate,
# ~71 TFLOP/s with fp16 accumulate.  Real transformer inference lands well
# under that, so we treat 71 as a hard physical ceiling for the sanity check.
GPU_PEAK_TFLOPS = 71.0


def cfg_params(config_path):
    """Rough non-embedding param count from config.json (12*L*H^2 transformer)."""
    try:
        with open(config_path, encoding="utf-8") as f:
            c = json.load(f)
    except OSError:
        return None, None
    H = c.get("hidden_size") or c.get("d_model")
    L = c.get("num_hidden_layers") or c.get("num_layers")
    V = c.get("vocab_size")
    if not H or not L:
        return None, None
    core = 12 * L * H * H
    emb = (V or 0) * H
    return core + emb, H


def integrity(name, X):
    Xf = np.asarray(X[: min(len(X), 20000)], dtype=np.float32)
    n_nan = int(np.isnan(Xf).sum())
    n_inf = int(np.isinf(Xf).sum())
    norms = np.linalg.norm(Xf, axis=1)
    zero_rows = int((norms == 0).sum())
    print("[%-10s] shape=%s dtype=%s | NaN=%d Inf=%d zero-rows=%d | "
          "row-norm min/med/max=%.3g/%.3g/%.3g"
          % (name, X.shape, X.dtype, n_nan, n_inf, zero_rows,
             norms.min(), np.median(norms), norms.max()), flush=True)
    ok = n_nan == 0 and n_inf == 0 and zero_rows == 0
    print("             -> %s" % ("OK" if ok else "*** FAIL: unusable ***"), flush=True)
    return ok


def l2(X):
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def cosine_stats(name, V):
    """Off-diagonal pairwise cosine distribution on an L2-normalised sample."""
    S = V @ V.T
    iu = np.triu_indices(len(V), k=1)
    s = S[iu]
    dup = float((s > 0.9999).mean())
    print("[%-10s] pairwise cosine (n=%d, %d pairs): mean=%.4f std=%.4f "
          "min=%.4f p50=%.4f max=%.4f | frac>0.9999=%.4f"
          % (name, len(V), len(s), s.mean(), s.std(), s.min(),
             np.median(s), s.max(), dup), flush=True)
    if s.std() < 1e-4:
        print("             -> *** COLLAPSE: all vectors nearly identical ***",
              flush=True)
        return None
    return s


def tanimoto_bits(B):
    """Tanimoto on 0/1 uint8 rows -> (n,n) float32."""
    Bf = np.asarray(B, dtype=np.float32)
    inter = Bf @ Bf.T
    pop = Bf.sum(axis=1)
    union = pop[:, None] + pop[None, :] - inter
    union[union == 0] = 1.0
    return inter / union


def spearman(x, y, cap=400_000):
    from scipy.stats import spearmanr
    if len(x) > cap:
        idx = RNG.choice(len(x), cap, replace=False)
        x, y = x[idx], y[idx]
    return float(spearmanr(x, y).statistic), len(x)


def agreement(labels, sims, ref, ref_name, mask=None, tag=""):
    """Spearman of each flat pair-similarity array against the flat reference.

    `mask` selects a subset of the pairs (used to drop Tanimoto's tie block at
    zero).  All arrays must already be flattened over the same triu index set.
    """
    if mask is not None:
        ref = ref[mask]
        sims = [s[mask] for s in sims]
    out = []
    for lab, s in zip(labels, sims):
        rho, npairs = spearman(s, ref)
        print("[agreement]%s %-9s vs %-16s Spearman, %7d pairs: rho=%+.4f"
              % (tag, lab, ref_name, npairs, rho), flush=True)
        out.append(rho)
    return out


def precision_at_k(S_learn, R_ref, k=10, queries=2000, name=""):
    """Tie-robust alternative to Spearman: retrieval precision of the top-k.

    For each query, what fraction of its k nearest neighbours under the learned
    cosine are also among its k nearest under Tanimoto?  Unlike a rank
    correlation this is unaffected by how many pairs tie at Tanimoto == 0, and
    1/k is the chance level for a random ranking in a sample of this size.
    """
    n = len(R_ref)
    L = S_learn.copy()
    T = R_ref.copy()
    np.fill_diagonal(L, -np.inf)
    np.fill_diagonal(T, -np.inf)
    qs = RNG.choice(n, min(queries, n), replace=False)
    kk = min(k, n - 1)
    li = np.argpartition(-L[qs], kk, axis=1)[:, :kk]
    ti = np.argpartition(-T[qs], kk, axis=1)[:, :kk]
    hits = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(li, ti)]
    p = float(np.mean(hits)) / kk
    print("[agreement] %-9s precision@%d vs Tanimoto top-%d: %.3f  (%d queries, "
          "chance ~%.3f)" % (name, kk, kk, p, len(qs), kk / max(n - 1, 1)),
          flush=True)
    return p


def neighbours(smiles, V, R, queries=3, k=3):
    """Print top-k neighbours under the learned cosine and under Tanimoto."""
    Sl = V @ V.T
    np.fill_diagonal(Sl, -np.inf)
    np.fill_diagonal(R, -np.inf)
    print("[neighbours] learned-encoder vs Morgan/Tanimoto top-%d, eyeball check:" % k,
          flush=True)
    for q in RNG.choice(len(smiles), min(queries, len(smiles)), replace=False):
        learn = np.argsort(-Sl[q])[:k]
        tani = np.argsort(-R[q])[:k]
        print("  Q: %s" % smiles[q], flush=True)
        for tag, idxs, mat in (("learned", learn, Sl), ("tanimoto", tani, R)):
            for r, j in enumerate(idxs, 1):
                print("     %-9s #%d  sim=%.4f  %s" % (tag, r, mat[q, j], smiles[j]),
                      flush=True)
        overlap = len(set(learn.tolist()) & set(tani.tolist()))
        print("     -> top-%d overlap = %d/%d" % (k, overlap, k), flush=True)


def throughput(name, meta, config_path, mol_per_s, max_len):
    n_params, H = cfg_params(config_path)
    L = meta["smiles"].astype(str).str.len()
    # ~1 token per character is the right order of magnitude for both SMILES
    # tokenizers (char-level for MoLFormer, byte-level BPE for ChemBERTa).
    tok = float(min(L.mean(), max_len))
    print("[%-10s] SMILES chars: mean=%.1f p50=%.0f p95=%.0f max=%.0f (max_len=%d)"
          % (name, L.mean(), L.median(), L.quantile(0.95), L.max(), max_len),
          flush=True)
    if not n_params:
        print("             -> config.json unreadable, skipping FLOP check", flush=True)
        return
    tok_per_s = mol_per_s * tok
    tflops = tok_per_s * 2.0 * n_params / 1e12
    verdict = "PLAUSIBLE" if tflops <= GPU_PEAK_TFLOPS else "*** EXCEEDS GPU PEAK ***"
    print("             params~%.1fM hidden=%d | %.0f mol/s x %.0f tok = %.0f tok/s "
          "-> ~%.1f TFLOP/s vs 3090 ceiling %.0f : %s (%.0f%% of peak)"
          % (n_params / 1e6, H, mol_per_s, tok, tok_per_s, tflops,
             GPU_PEAK_TFLOPS, verdict, 100 * tflops / GPU_PEAK_TFLOPS), flush=True)


def assert_aligned(tag, a, b, n):
    bad = int((a[:n].values != b[:n].values).sum())
    if bad:
        sys.exit("[validate] ERROR: %s rows are NOT aligned (%d/%d SMILES differ). "
                 "Cannot compare these embeddings." % (tag, bad, n))
    print("[align] %s: first %d SMILES identical across files -> row-aligned OK"
          % (tag, n), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molformer", default=os.path.join(EMB, "molformer/output/molformer_emb_merged.npy"))
    ap.add_argument("--molformer-meta", default=os.path.join(EMB, "molformer/output/molformer_smiles_merged.csv"))
    ap.add_argument("--chemberta", default=os.path.join(EMB, "chemberta2/output/chemberta2_emb_merged.npy"))
    ap.add_argument("--chemberta-meta", default=os.path.join(EMB, "chemberta2/output/chemberta2_smiles_merged.csv"))
    ap.add_argument("--morgan", default=os.path.join(EMB, "morgan_fp/output/morgan_emb.npy"))
    ap.add_argument("--morgan-meta", default=os.path.join(EMB, "morgan_fp/output/morgan_smiles.csv"))
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--molformer-rate", type=float, default=6733.0)
    ap.add_argument("--chemberta-rate", type=float, default=22122.0)
    args = ap.parse_args()

    for p in (args.molformer, args.chemberta, args.morgan):
        if not os.path.exists(p):
            sys.exit("[validate] ERROR: missing %s (run merge_shards.py first?)" % p)

    mf, mf_m = np.load(args.molformer, mmap_mode="r"), pd.read_csv(args.molformer_meta)
    cb, cb_m = np.load(args.chemberta, mmap_mode="r"), pd.read_csv(args.chemberta_meta)
    mg, mg_m = np.load(args.morgan, mmap_mode="r"), pd.read_csv(args.morgan_meta)

    n = min(len(mf), len(cb), len(mg), len(mf_m), len(cb_m), len(mg_m))
    print("=" * 78)
    print("[validate] rows available: molformer=%d chemberta=%d morgan=%d -> compare %d"
          % (len(mf), len(cb), len(mg), n), flush=True)
    assert_aligned("molformer/chemberta", mf_m["smiles"], cb_m["smiles"], min(n, 5000))
    assert_aligned("molformer/morgan", mf_m["smiles"], mg_m["smiles"], min(n, 5000))

    print("=" * 78)
    ok = all([integrity("molformer", mf), integrity("chemberta", cb),
              integrity("morgan", mg)])

    idx = np.sort(RNG.choice(n, min(args.sample, n), replace=False))
    smiles = mf_m["smiles"].astype(str).values[idx]
    V_mf = l2(mf[idx])
    V_cb = l2(cb[idx])
    V_mg = np.asarray(mg[idx], dtype=np.uint8)

    print("=" * 78)
    s_mf = cosine_stats("molformer", V_mf)
    s_cb = cosine_stats("chemberta", V_cb)
    R = tanimoto_bits(V_mg)
    iu = np.triu_indices(len(R), k=1)
    r = R[iu]
    print("[%-10s] Tanimoto (reference): mean=%.4f std=%.4f min=%.4f p50=%.4f max=%.4f"
          % ("morgan", r.mean(), r.std(), r.min(), np.median(r), r.max()), flush=True)
    zero_frac = float((r == 0).mean())
    print("[%-10s] Tanimoto ties: frac==0 %.4f, frac<0.05 %.4f, %d distinct values"
          " -> Spearman over ALL pairs is attenuated; read the >0 column too"
          % ("morgan", zero_frac, float((r < 0.05).mean()), len(np.unique(r))),
          flush=True)

    print("=" * 78)
    if s_mf is None or s_cb is None:
        print("[validate] STOP: collapsed embeddings, semantic checks meaningless",
              flush=True)
        return 1
    S_mf = V_mf @ V_mf.T
    S_cb = V_cb @ V_cb.T
    f_mf, f_cb, f_ref = S_mf[iu], S_cb[iu], r
    nz = f_ref > 0
    print("            -- all %d pairs (includes the %.0f%% Tanimoto==0 tie block) --"
          % (len(f_ref), 100 * zero_frac), flush=True)
    agreement(["molformer", "chemberta"], [f_mf, f_cb], f_ref, "morgan/tanimoto")
    rho_ab = spearman(f_mf, f_cb)[0]
    print("[agreement] molformer vs chemberta               : rho=%+.4f" % rho_ab, flush=True)
    print("            -- only the %d pairs with Tanimoto > 0 (tie-free) --"
          % int(nz.sum()), flush=True)
    agreement(["molformer", "chemberta"], [f_mf, f_cb], f_ref, "morgan/tanimoto",
              mask=nz, tag="[>0]")
    print("[agreement] molformer vs chemberta  [>0]         : rho=%+.4f"
          % spearman(f_mf[nz], f_cb[nz])[0], flush=True)
    precision_at_k(S_mf, R, k=10, name="molformer")
    precision_at_k(S_cb, R, k=10, name="chemberta")

    print("=" * 78)
    neighbours(smiles, V_mf, R)

    print("=" * 78)
    throughput("molformer", mf_m, os.path.join(EMB, "molformer/code/MoLFormer-XL-both-10pct/config.json"),
               args.molformer_rate, 202)
    throughput("chemberta", cb_m, os.path.join(EMB, "chemberta2/code/ChemBERTa-77M-MTR/config.json"),
               args.chemberta_rate, 256)

    print("=" * 78)
    print("[validate] integrity=%s | done" % ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
