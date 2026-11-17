# -*- coding: utf-8 -*-
"""ChemBERTa-2 (77M-MTR) embedding for PCQM4Mv2 SMILES.  (384-d, [CLS])

Input : /mnt/data/yangsensen/data/*.jsonl.gz  (or smiles_all.parquet cache)
Output: ../output/chemberta2_emb_shardXofY.npy   (N, 384) float16
        ../output/chemberta2_smiles_shardXofY.csv
Usage : CUDA_VISIBLE_DEVICES=1 python embed_chemberta2.py --shard 1 --num-shards 2
        python embed_chemberta2.py --limit 2000          # smoke test

Two notes on numerics.

1. The checkpoint loads from model.safetensors, converted from pytorch_model.bin
   because transformers >= 4.57 refuses torch.load on .bin with torch < 2.6
   (CVE-2025-32434). The original .bin is kept untouched.

2. Compute dtype defaults to bfloat16, matching embed_molformer.py. Unlike
   MoLFormer, this model uses softmax attention so its intermediates are bounded
   by construction and float16 is genuinely safe here -- a 50k-molecule run came
   out clean. bf16 is still the default because the full run is 3.45M molecules,
   70x longer: one pathological SMILES overflowing fp16 would abort the whole
   shard and cost hours, whereas bf16 keeps fp32's exponent range at the same
   tensor-core speed. The precision cost (cosine ~0.9999 vs fp32) is irrelevant
   to UMAP + HDBSCAN. Pass --dtype float16 if you need the extra mantissa bits.

Either way the loop below counts non-finite values and values too large for the
fp16 store, and refuses to write output if it sees any.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(BASE, "..", "..")))   # mol_io.py lives in embedding/

from mol_io import (DEFAULT_DTYPE, DTYPES, assert_finite,             # noqa: E402
                    load_low_precision_model, load_smiles)

OUTDIR = os.path.normpath(os.path.join(BASE, "..", "output"))
MODEL_PATH = os.path.join(BASE, "ChemBERTa-77M-MTR")

# Drain the non-finite counter this often (in batches). Checking every batch
# would force a GPU sync and serialise the pipeline; every 20 batches still
# catches a broken run within seconds instead of after the whole shard.
CHECK_EVERY = 20
FP16_STORE_MAX = 65504.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only embed first N molecules (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default=DEFAULT_DTYPE, choices=sorted(DTYPES),
                    help="compute dtype (default %(default)s). float16 is safe "
                         "for this softmax-attention model; bfloat16 just removes "
                         "the tail risk on a 3.45M-molecule run.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(OUTDIR, exist_ok=True)

    df = load_smiles()
    # Shard FIRST, then truncate, so that with --limit L the shards together
    # cover exactly the first L*num_shards parquet rows in order. That ordering
    # is what merge_shards.py relies on to restore the full row sequence.
    df = df.iloc[args.shard:: args.num_shards].reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
    n = len(df)
    print("[chemberta2] shard %d/%d: %d molecules, bs=%d, dtype=%s"
          % (args.shard, args.num_shards, n, args.batch_size, args.dtype), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = load_low_precision_model(MODEL_PATH, dtype=args.dtype)
    hidden = model.config.hidden_size
    print("[chemberta2] model loaded, hidden=%d" % hidden, flush=True)

    emb = np.empty((n, hidden), dtype=np.float16)
    bad = torch.zeros((), dtype=torch.long, device="cuda")
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            batch = df["smiles"].iloc[i: i + args.batch_size].tolist()
            enc = tok(batch, padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to("cuda")
            out = model(**enc).last_hidden_state[:, 0, :]
            vec = out.float()
            # Counted on the GPU so the hot loop never syncs; drained below.
            bad += ((~torch.isfinite(vec)) | (vec.abs() > FP16_STORE_MAX)).sum()
            emb[i: i + len(vec)] = vec.cpu().numpy().astype(np.float16)
            done += len(vec)
            if done % (args.batch_size * CHECK_EVERY) < args.batch_size:
                if int(bad) > 0:
                    raise SystemExit(
                        "[chemberta2] FATAL: %d non-finite / fp16-overflowing values "
                        "after %d rows (dtype=%s). Refusing to continue; re-run with "
                        "--dtype bfloat16 or float32." % (int(bad), done, args.dtype))
            if done % (args.batch_size * 20) < args.batch_size or done == n:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print("  %d/%d  %.0f mol/s  elapsed %.0fs  eta %.0fs"
                      % (done, n, rate, el, (n - done) / max(rate, 1e-9)), flush=True)

    if int(bad) > 0:
        raise SystemExit("[chemberta2] FATAL: %d non-finite / overflowing values in the "
                         "full shard (dtype=%s)." % (int(bad), args.dtype))
    # Belt and braces: validate the array we are actually about to persist.
    assert_finite(emb, "chemberta2 shard%dof%d embedding" % (args.shard, args.num_shards))

    tag = "shard%dof%d" % (args.shard, args.num_shards)
    np.save(os.path.join(OUTDIR, "chemberta2_emb_%s.npy" % tag), emb)
    df.to_csv(os.path.join(OUTDIR, "chemberta2_smiles_%s.csv" % tag), index=False)
    E = emb.astype(np.float32)
    print("[chemberta2] saved %s -> %s  (%.0f mol/s)  finite=OK  "
          "|v| max=%.4g mean=%.4g std=%.4g"
          % (emb.shape, OUTDIR, n / max(time.time() - t0, 1e-9),
             np.abs(E).max(), np.abs(E).mean(), E.std()), flush=True)


if __name__ == "__main__":
    main()
