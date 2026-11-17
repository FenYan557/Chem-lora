#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pin down why MoLFormer emits 100% NaN, and pick a dtype that does not.

What we already know from a first pass
--------------------------------------
  fp32 -> finite, bf16 -> finite, fp16 -> every value NaN, first non-finite
  module = encoder.layer.0.attention.self.

  Raising linear_attention_eps from 1e-6 to 1e-2 did NOT help, which rules out
  the obvious guess ("dividing by a tiny clamped denominator").  The NaN must
  therefore enter through an intermediate that leaves the fp16 range on its own
  -- fp16 tops out at 65504 while bf16 shares fp32's exponent range, which is
  exactly why bf16 survives and fp16 does not.

So this script stops guessing and measures.  probe_attention() replays layer 0's
linear attention one matmul at a time and reports the magnitude of every
intermediate alongside its inf/NaN count, so the first step that leaves the
dtype's range is named explicitly rather than inferred.

Also covered:
  * mask_check  - the compat shim's additive mask really is recovered as 1/0 by
                  the model's `(mask == 0)` conversion (it compares -0.0 for
                  allowed positions, which is fine under IEEE equality).
  * dtype_fidelity - bf16 vs fp32 [CLS] cosine, to justify using bf16.
  * timed - throughput per dtype, so the fix does not silently cost 10x.
"""
import os
import sys
import time

import torch

CODE = "/mnt/data/yangsensen/embedding/molformer/code"
EMB = "/mnt/data/yangsensen/embedding"
MODEL_PATH = os.path.join(CODE, "MoLFormer-XL-both-10pct")
sys.path.insert(0, CODE)
sys.path.insert(0, EMB)

import molformer_compat                            # noqa: E402
molformer_compat.install()
from mol_io import load_smiles                     # noqa: E402
from transformers import AutoModel, AutoTokenizer   # noqa: E402

BATCH = 64
MAXLEN = 202


def load_model(dtype, eps=None):
    """fp32 build on CPU (no half QR on CPU) -> cast -> GPU, deterministic fmap."""
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    fixed = 0
    for m in model.modules():
        if m.__class__.__name__.endswith("FeatureMap") and hasattr(m, "deterministic"):
            if not m.deterministic:
                m.deterministic = True
                fixed += 1
    if fixed != 12:
        print("  !! expected 12 feature maps, forced %d" % fixed, flush=True)
    if eps is not None:
        n = 0
        for m in model.modules():
            if hasattr(m, "eps") and m.__class__.__name__.endswith("SelfAttention"):
                m.eps = eps
                n += 1
        print("  (patched eps=%g on %d attention modules)" % (eps, n), flush=True)
    return model.to(dtype).cuda().eval()


def encode(smiles, tok):
    enc = tok(smiles, padding=True, truncation=True, max_length=MAXLEN,
              return_tensors="pt").to("cuda")
    return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def first_nonfinite(model, enc):
    hits = []

    def mk(name):
        def hook(_mod, _inp, out):
            t = out if torch.is_tensor(out) else (
                out[0] if isinstance(out, (tuple, list)) and len(out)
                and torch.is_tensor(out[0]) else None)
            if t is not None and not bool(torch.isfinite(t).all()):
                hits.append(name)
        return hook

    handles = [m.register_forward_hook(mk(n)) for n, m in model.named_modules()]
    try:
        with torch.no_grad():
            out = model(**enc).last_hidden_state
    finally:
        for h in handles:
            h.remove()
    return (hits[0] if hits else None), len(hits), out


def run(tag, dtype, enc, eps=None):
    model = load_model(dtype, eps=eps)
    first, nhits, out = first_nonfinite(model, enc)
    o = out.float()
    n_nan = int(torch.isnan(o).sum())
    print("[%s] dtype=%-9s eps=%-6s -> out%s  NaN=%d/%d  %s"
          % (tag, str(dtype).replace("torch.", ""), eps or "1e-6",
             tuple(out.shape), n_nan, o.numel(),
             "FINITE" if n_nan == 0 else "*** NaN ***"), flush=True)
    if n_nan == 0:
        print("        |out| max=%.4g mean=%.4g std=%.4g"
              % (o.abs().max(), o.abs().mean(), o.std()), flush=True)
    if first:
        print("        first non-finite module: %s  (%d modules affected)"
              % (first, nhits), flush=True)
    del model
    torch.cuda.empty_cache()
    return n_nan == 0


def mask_check(enc, dtype):
    am = enc["attention_mask"]
    ext = molformer_compat.create_bidirectional_mask(
        config=None,
        inputs_embeds=torch.zeros(am.shape[0], am.shape[1], 8, dtype=dtype,
                                 device=am.device),
        attention_mask=am)
    back = (ext == 0).to(ext.dtype)
    # ext is 4-D (B, 1, 1, KV): squeeze BOTH broadcast dims before comparing.
    ok = bool(torch.equal(back[:, 0, 0, :], am.to(back.dtype)))
    print("[mask] shim additive mask: shape=%s dtype=%s min=%.4g max=%.4g"
          % (tuple(ext.shape), ext.dtype, float(ext.min()), float(ext.max())), flush=True)
    print("[mask] model's (mask==0) recovers the original 1/0 padding mask: %s"
          % ("YES" if ok else "*** NO ***"), flush=True)
    print("[mask] allowed positions encode as -0.0 -> harmless, IEEE -0.0 == 0.0 "
          "is True", flush=True)
    print("[mask] finfo(%s).min=%.4g   fp16 max finite=65504"
          % (str(dtype).replace("torch.", ""), torch.finfo(dtype).min), flush=True)
    return ok


def probe_attention(dtype, smiles, tok, eps=None, layer=0):
    """Replay one layer's linear attention step by step; name the first overflow."""
    model = load_model(dtype, eps=eps)
    attn = model.encoder.layer[layer].attention.self
    mod = sys.modules[type(attn).__module__]   # HF dynamic module: not importable by name
    apply_rotary = mod.apply_rotary_pos_emb
    enc = encode(smiles, tok)

    with torch.no_grad():
        emb = model.embeddings(input_ids=enc["input_ids"])
        am = molformer_compat.create_bidirectional_mask(
            config=model.config, inputs_embeds=emb,
            attention_mask=enc["attention_mask"])
        pos = torch.arange(emb.shape[1], device=emb.device).unsqueeze(0)

        q = attn.transpose_for_scores(attn.query(emb))
        k = attn.transpose_for_scores(attn.key(emb))
        v = attn.transpose_for_scores(attn.value(emb))
        kv = k.shape[-2]
        cos, sin = attn.rotary_embeddings(v, seq_len=kv)
        q, k = apply_rotary(q, k, cos, sin, pos)
        q, k = attn.feature_map(q, k)
        am01 = (am == 0).to(am.dtype)
        k = k * am01[:, 0, -1][:, None, -kv:, None]

        key_value = torch.matmul(k.transpose(-1, -2), v)
        numer = torch.matmul(q, key_value)
        norm = torch.matmul(q, k.sum(dim=-2).unsqueeze(-1))
        ctx = numer / norm.clamp(min=attn.eps)

    fmax = torch.finfo(dtype).max
    print("[probe] dtype=%-9s eps=%-6g fp-max=%.4g  (layer %d)"
          % (str(dtype).replace("torch.", ""), attn.eps, fmax, layer), flush=True)
    steps = [("embedding", emb),
             ("q  (post feature-map ReLU)", q),
             ("k  (post fmap, mask-mult)", k),
             ("v", v),
             ("key_value = k^T @ v", key_value),
             ("NUMERATOR = q @ key_value", numer),
             ("norm (raw, pre-clamp)", norm),
             ("context = numer / clamp(norm)", ctx)]
    culprit = None
    for name, t in steps:
        tf = t.float()
        n_inf = int(torch.isinf(tf).sum())
        n_nan = int(torch.isnan(tf).sum())
        flag = ""
        if n_nan:
            flag = "   <== NaN"
        elif n_inf:
            flag = "   <== INF, exceeds %.4g" % fmax
        if flag and culprit is None:
            culprit = name
        print("        %-32s max|.|=%-13.6g min=%-12.6g inf=%-7d nan=%-7d%s"
              % (name, float(tf.abs().max()), float(tf.min()), n_inf, n_nan, flag),
              flush=True)
    print("        -> first step out of range: %s" % (culprit or "none"), flush=True)
    del model
    torch.cuda.empty_cache()
    return culprit


def dtype_fidelity(smiles, tok):
    """bf16 vs fp32 on identical input: are the [CLS] vectors interchangeable?"""
    enc = encode(smiles, tok)
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        m = load_model(dt)
        with torch.no_grad():
            outs[dt] = m(**enc).last_hidden_state[:, 0, :].float().cpu()
        del m
        torch.cuda.empty_cache()
    a, b = outs[torch.float32], outs[torch.bfloat16]
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    print("[fidelity] bf16 vs fp32 [CLS]: cosine min=%.6f mean=%.6f | max abs diff=%.4g "
          "| rel L2 err=%.3g"
          % (float(cos.min()), float(cos.mean()), float((a - b).abs().max()),
             float((a - b).norm() / a.norm())), flush=True)
    return float(cos.min())


def timed(dtype, smiles, tok, rounds=5):
    model = load_model(dtype)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        encode(smiles, tok)
        model(**encode(smiles, tok))          # warmup: CUDA kernels / autotune
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(rounds):
            model(**encode(smiles, tok))
        torch.cuda.synchronize()
        dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print("[timing] %-9s %d mols x %d rounds in %.2fs -> %6.0f mol/s   peak GPU %.0f MiB"
          % (str(dtype).replace("torch.", ""), len(smiles), rounds, dt,
             len(smiles) * rounds / dt, peak), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    smiles = load_smiles()["smiles"].tolist()[:BATCH]
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    enc = encode(smiles, tok)

    print("=" * 96, flush=True)
    print("[setup] batch=%d seq_len=%d padding present=%s"
          % (BATCH, enc["input_ids"].shape[1],
             bool((enc["attention_mask"] == 0).any())), flush=True)

    print("=" * 96, flush=True)
    mask_check(enc, torch.float16)

    print("=" * 96, flush=True)
    ok32 = run("A", torch.float32, enc)
    okbf = run("A", torch.bfloat16, enc)
    ok16 = run("A", torch.float16, enc)

    print("=" * 96, flush=True)
    ok16_eps = run("C", torch.float16, enc, eps=1e-2)

    print("=" * 96, flush=True)
    probe_attention(torch.float32, smiles, tok)
    print("-" * 96, flush=True)
    probe_attention(torch.float16, smiles, tok)
    print("-" * 96, flush=True)
    probe_attention(torch.float16, smiles, tok, eps=1e-2)
    print("-" * 96, flush=True)
    probe_attention(torch.bfloat16, smiles, tok)

    print("=" * 96, flush=True)
    dtype_fidelity(smiles, tok)

    print("=" * 96, flush=True)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        timed(dt, smiles, tok)

    print("=" * 96, flush=True)
    print("[verdict] fp32 finite=%s | bf16 finite=%s | fp16 finite=%s | "
          "fp16+eps1e-2 finite=%s" % (ok32, okbf, ok16, ok16_eps), flush=True)
    if ok32 and okbf and not ok16:
        print("[verdict] CONFIRMED: MoLFormer linear attention leaves the fp16 range. "
              "bf16/fp32 are safe; fp16 must NOT be used for this model.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
