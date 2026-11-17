# -*- coding: utf-8 -*-
"""MoLFormer-XL embedding for PCQM4Mv2 SMILES.  (768-d, [CLS])

Input : /mnt/data/yangsensen/data/*.jsonl.gz  (or smiles_all.parquet cache)
Output: ../output/molformer_emb_shardXofY.npy   (N, 768) float16
        ../output/molformer_smiles_shardXofY.csv
Usage : CUDA_VISIBLE_DEVICES=0 python embed_molformer.py --shard 0 --num-shards 2
        python embed_molformer.py --limit 2000          # smoke test

Compute dtype defaults to bfloat16, NOT float16.  MoLFormer's linear attention
has no softmax, so its `q @ (k^T @ v)` numerator reaches ~1.4e6 while fp16 tops
out at 65504; in float16 this overflows to inf and every output vector becomes
NaN.  See mol_io.DTYPES for the measurements.  Storage stays float16 because the
[CLS] values themselves are small (max ~14), and the loop below aborts rather
than saving if anything is non-finite or would overflow the fp16 store.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.normpath(os.path.join(BASE, "..", "..")))   # mol_io.py lives in embedding/

import molformer_compat          # noqa: E402  must run before the model's remote code is imported
from mol_io import (DEFAULT_DTYPE, DTYPES, assert_finite,             # noqa: E402
                    load_low_precision_model, load_smiles)

OUTDIR = os.path.normpath(os.path.join(BASE, "..", "output"))
MODEL_PATH = os.path.join(BASE, "MoLFormer-XL-both-10pct")

# Drain the non-finite counter this often (in batches). Checking every batch
# would force a GPU sync and serialise the pipeline; every 20 batches still
# catches a broken run within seconds instead of after the whole shard.
CHECK_EVERY = 20
FP16_STORE_MAX = 65504.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=202)
    ap.add_argument("--limit", type=int, default=0, help="only embed first N molecules (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default=DEFAULT_DTYPE, choices=sorted(DTYPES),
                    help="compute dtype (default %(default)s). float16 produces "
                         "all-NaN for this model -- do not use it.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(OUTDIR, exist_ok=True)
    molformer_compat.install()

    df = load_smiles()
    # Shard FIRST, then truncate: shard i holds df.iloc[i::num_shards], so with
    # --limit L the shards together cover exactly the first L*num_shards rows of
    # the parquet in order. merge_shards.py relies on this to restore row order.
    df = df.iloc[args.shard:: args.num_shards].reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
    n = len(df)
    print("[molformer] shard %d/%d: %d molecules, bs=%d, dtype=%s"
          % (args.shard, args.num_shards, n, args.batch_size, args.dtype), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = load_low_precision_model(MODEL_PATH, trust_remote_code=True, dtype=args.dtype)
    hidden = model.config.hidden_size
    print("[molformer] model loaded, hidden=%d" % hidden, flush=True)

    emb = np.empty((n, hidden), dtype=np.float16)
    bad = torch.zeros((), dtype=torch.long, device="cuda")
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch = df["smiles"].iloc[i: i + args.batch_size].tolist()
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to("cuda")
            # MolformerModel.forward() only accepts input_ids / attention_mask;
            # the tokenizer also returns token_type_ids, which raises TypeError.
            enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            out = model(**enc).last_hidden_state[:, 0, :]
            vec = out.float()
            # Counted on the GPU so the hot loop never syncs; drained below.
            bad += ((~torch.isfinite(vec)) | (vec.abs() > FP16_STORE_MAX)).sum()
            emb[i: i + len(vec)] = vec.cpu().numpy().astype(np.float16)
            done += len(vec)
            if done % (args.batch_size * CHECK_EVERY) < args.batch_size:
                if int(bad) > 0:
                    raise SystemExit(
                        "[molformer] FATAL: %d non-finite / fp16-overflowing values "
                        "after %d rows (dtype=%s). Refusing to continue. Use "
                        "--dtype bfloat16 (or float32); float16 overflows this "
                        "model's linear-attention numerator."
                        % (int(bad), done, args.dtype))
            if done % (args.batch_size * 20) < args.batch_size or done == n:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (n - done) / max(rate, 1e-9)
                print("  %d/%d  %.0f mol/s  elapsed %.0fs  eta %.0fs"
                      % (done, n, rate, el, eta), flush=True)

    if int(bad) > 0:
        raise SystemExit("[molformer] FATAL: %d non-finite / overflowing values in the "
                         "full shard (dtype=%s)." % (int(bad), args.dtype))
    # Belt and braces: validate the array we are actually about to persist.
    assert_finite(emb, "molformer shard%dof%d embedding" % (args.shard, args.num_shards))

    tag = "shard%dof%d" % (args.shard, args.num_shards)
    np.save(os.path.join(OUTDIR, "molformer_emb_%s.npy" % tag), emb)
    df.to_csv(os.path.join(OUTDIR, "molformer_smiles_%s.csv" % tag), index=False)
    E = emb.astype(np.float32)
    print("[molformer] saved %s -> %s  (%.0f mol/s)  finite=OK  "
          "|v| max=%.4g mean=%.4g std=%.4g"
          % (emb.shape, OUTDIR, n / max(time.time() - t0, 1e-9),
             np.abs(E).max(), np.abs(E).mean(), E.std()), flush=True)


if __name__ == "__main__":
    main()
