#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Locate every NaN/Inf in the MoLFormer / ChemBERTa / Morgan artefacts.

merge_shards.py reported the merged MoLFormer array as entirely NaN.  This
script answers three questions the handoff doc cannot:
  * is it the merge, or were the shard files already NaN?
  * was the ORIGINAL 1k-2k smoke test (parked in ../tmp/smoke_out) also NaN,
    i.e. has this been broken since before the handoff?
  * which rows / how many are affected, and are ChemBERTa + Morgan clean?
"""
import os
import sys

import numpy as np

B = "/mnt/data/yangsensen"
CANDIDATES = [
    ("molformer SMOKE (pre-handoff)", B + "/tmp/smoke_out/molformer_emb_shard0of1.npy"),
    ("molformer shard0of2",           B + "/embedding/molformer/output/molformer_emb_shard0of2.npy"),
    ("molformer shard1of2",           B + "/embedding/molformer/output/molformer_emb_shard1of2.npy"),
    ("molformer merged",              B + "/embedding/molformer/output/molformer_emb_merged.npy"),
    ("chemberta SMOKE (pre-handoff)", B + "/tmp/smoke_out/chemberta2_emb_shard0of1.npy"),
    ("chemberta shard0of2",           B + "/embedding/chemberta2/output/chemberta2_emb_shard0of2.npy"),
    ("chemberta merged",              B + "/embedding/chemberta2/output/chemberta2_emb_merged.npy"),
    ("morgan SMOKE (pre-handoff)",    B + "/tmp/smoke_out/morgan_emb.npy"),
    ("morgan 50k",                    B + "/embedding/morgan_fp/output/morgan_emb.npy"),
]


def scan(tag, path, rowcap=200000):
    if not os.path.exists(path):
        print("[%-30s] MISSING  %s" % (tag, path), flush=True)
        return None
    X = np.load(path, mmap_mode="r")
    n = min(len(X), rowcap)
    Xf = np.asarray(X[:n], dtype=np.float32)
    nan = np.isnan(Xf)
    inf = np.isinf(Xf)
    tot_nan, tot_inf = int(nan.sum()), int(inf.sum())
    per_row = nan.any(axis=1) | inf.any(axis=1)
    n_bad_rows = int(per_row.sum())
    finite = Xf[~(nan | inf)]
    if finite.size:
        stat = "|finite| max=%.4g mean=%.4g  std=%.4g" % (
            np.abs(finite).max(), np.abs(finite).mean(), Xf.std())
    else:
        stat = "no finite values at all"
    verdict = "CLEAN" if tot_nan + tot_inf == 0 else "*** NaN/Inf ***"
    print("[%-30s] %-22s rows=%d/%d bad_rows=%d (%.1f%%) NaN=%d Inf=%d  %s  %s"
          % (tag, os.path.basename(path), n, len(X), n_bad_rows,
             100.0 * n_bad_rows / max(n, 1), tot_nan, tot_inf, verdict, stat),
          flush=True)
    if n_bad_rows and n_bad_rows < n:
        first = int(np.argmax(per_row))
        last = int(n - 1 - np.argmax(per_row[::-1]))
        print("%34s first bad row=%d last bad row=%d" % ("", first, last), flush=True)
    return tot_nan + tot_inf


def main():
    print("=" * 100, flush=True)
    bad = 0
    for tag, path in CANDIDATES:
        r = scan(tag, path)
        if r:
            bad += 1
    print("=" * 100, flush=True)
    print("[nan_check] %d artefact(s) contain NaN/Inf" % bad, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
