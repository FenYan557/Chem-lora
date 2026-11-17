# -*- coding: utf-8 -*-
"""Shared SMILES loading + model loading helpers for all three embedders.

Reads the parquet cache built by build_smiles_cache.py when present, otherwise
parses the raw *.jsonl.gz (SMILES live at labels.object.SMILES).
Also centralises the low-precision loading quirks of this env.
"""
import glob
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import torch

DATA = "/mnt/data/yangsensen/data"
CACHE = os.path.join(DATA, "smiles_all.parquet")

# ---------------------------------------------------------------------------
# Compute dtype
# ---------------------------------------------------------------------------
# bfloat16 is the default and is NOT interchangeable with float16 here.
#
# MoLFormer uses Performer-style linear attention with a ReLU feature map and no
# softmax, so nothing bounds its intermediates. Measured on layer 0 with a real
# 64-molecule batch:
#
#     q (post feature-map)   max|.| = 1.3e2
#     k (post feature-map)   max|.| = 2.1e2
#     key_value = k^T @ v    max|.| = 6.1e3
#     NUMERATOR = q @ key_value   max|.| = 1.4e6   <-- fp16 ceiling is 65504
#     norm = q @ k.sum(-2)        max|.| = 3.4e5
#     context = NUMERATOR / norm.clamp(eps)  max|.| = 6.7
#
# In float16 that numerator overflows to inf, the norm overflows to inf too, so
# context = inf/inf = NaN, and LayerNorm spreads NaN to every position of every
# row. The script still saved a well-shaped .npy and still printed a healthy
# mol/s -- a completely silent failure.
#
# Note the raw `norm` bottoms out around 7e3, far above linear_attention_eps
# (1e-6), so the clamp never engages: raising eps does NOT fix this. The problem
# is the numerator's exponent range, which is exactly what bfloat16 restores
# (max 3.4e38, same as fp32) at the same tensor-core speed on Ampere.
#
# Measured on RTX 3090, batch 64: fp32 2979 mol/s, bf16 6389 mol/s,
# fp16 6224 mol/s (but all-NaN). bf16 vs fp32 [CLS]: cosine min 0.999908,
# rel L2 err 9.5e-3.
#
# float16 remains available for softmax-attention models such as ChemBERTa,
# whose attention output is bounded by construction.
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
DEFAULT_DTYPE = "bfloat16"


def load_smiles(data_dir=DATA, cache=CACHE):
    """-> DataFrame[smiles, split] covering every molecule in the dataset."""
    if cache and os.path.exists(cache):
        t0 = time.time()
        df = pd.read_parquet(cache)
        print("[mol_io] loaded cache %s rows=%d in %.1fs" % (cache, len(df), time.time() - t0),
              flush=True)
        return df

    print("[mol_io] no parquet cache, parsing gz (run build_smiles_cache.py once to speed this up)",
          flush=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.jsonl.gz"))):
        split = "valid" if "valid" in os.path.basename(f) else "train"
        with gzip.open(f, "rt") as fh:
            for line in fh:
                try:
                    smi = json.loads(line)["labels"]["object"]["SMILES"]
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append((smi, split))
    return pd.DataFrame(rows, columns=["smiles", "split"])


def resolve_dtype(name):
    """-> torch.dtype for a key of DTYPES (a torch.dtype is passed through)."""
    if isinstance(name, torch.dtype):
        return name
    key = str(name).lower().replace("torch.", "")
    if key in ("bf16", "bfloat"):
        key = "bfloat16"
    if key in ("fp16", "half"):
        key = "float16"
    if key in ("fp32", "single", "float"):
        key = "float32"
    if key not in DTYPES:
        raise SystemExit("[mol_io] unknown dtype %r, expected one of %s"
                         % (name, sorted(DTYPES)))
    return DTYPES[key]


def load_low_precision_model(path, trust_remote_code=False, dtype=DEFAULT_DTYPE):
    """Build in fp32 on CPU, then cast to `dtype` and move to the GPU.

    Two independent traps motivate building on the CPU first:

    * Passing dtype/torch_dtype=float16 straight to from_pretrained breaks
      MoLFormer: its MolformerFeatureMap calls torch.linalg.qr during __init__,
      and CPU QR is not implemented for Half ("geqrf_cpu" not implemented).
      Building in fp32 and casting afterwards avoids that and keeps the same
      low-precision inference speed/memory on the GPU.
    * MoLFormer's config ships deterministic_eval=false, so every forward pass
      would rebuild the random Fourier projection via torch.randn in fp32. That
      overwrites the cast buffers ("expected scalar type Float but found Half")
      and, worse, gives each batch a different projection so the embeddings are
      not reproducible. We force the deterministic branch, which uses the
      feature_map.weight already stored in the checkpoint (all 12 layers).
    """
    from transformers import AutoModel

    dtype = resolve_dtype(dtype)
    model = AutoModel.from_pretrained(path, trust_remote_code=trust_remote_code)

    fixed = 0
    for m in model.modules():
        if m.__class__.__name__.endswith("FeatureMap") and hasattr(m, "deterministic"):
            if not m.deterministic:
                m.deterministic = True
                fixed += 1
    if fixed:
        print("[mol_io] forced deterministic feature map on %d module(s) "
              "-> reproducible embeddings" % fixed, flush=True)

    print("[mol_io] %s -> %s on GPU" % (os.path.basename(os.path.normpath(path)), dtype),
          flush=True)
    return model.to(dtype).cuda().eval()


def load_fp16_model(path, trust_remote_code=False, dtype=DEFAULT_DTYPE):
    """Back-compatible alias. NOTE: no longer forces float16 -- it honours
    DEFAULT_DTYPE (bfloat16), because float16 yields all-NaN for MoLFormer."""
    return load_low_precision_model(path, trust_remote_code=trust_remote_code, dtype=dtype)


def assert_finite(arr, what, rows=None):
    """Abort loudly on NaN/Inf instead of persisting a garbage .npy.

    The whole reason an all-NaN MoLFormer embedding survived a "smoke test
    passed" handoff is that nothing ever looked at the values -- only at the
    shape. Every embedder must call this before saving.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8 or np.issubdtype(a.dtype, np.integer):
        return a
    f = a.astype(np.float32, copy=False)
    n_nan = int(np.isnan(f).sum())
    n_inf = int(np.isinf(f).sum())
    if n_nan or n_inf:
        where = "" if rows is None else " at rows %s" % (rows,)
        raise SystemExit(
            "[mol_io] FATAL: %s contains NaN=%d Inf=%d%s -- refusing to write "
            "output. For MoLFormer the compute dtype must be bfloat16 or "
            "float32: float16 overflows its linear-attention numerator "
            "(measured max ~1.4e6) past the fp16 limit of 65504."
            % (what, n_nan, n_inf, where))
    return a
