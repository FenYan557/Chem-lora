# -*- coding: utf-8 -*-
"""Compat shim so MoLFormer's HF modeling code runs on transformers 4.57.x.

Why this exists
---------------
The current HF revision of IBM/MoLFormer-XL-both-10pct does::

    from transformers.masking_utils import create_bidirectional_mask

`transformers.masking_utils` exists in 4.56+, but `create_bidirectional_mask`
was only added in transformers 5.x. transformers 5.x in turn hard-requires
torch >= 2.6, and this box runs torch 2.5.1+cu121 because RAPIDS cuml-cu12
26.10 (GPU UMAP/HDBSCAN, already verified working) pulled in CUDA 12.9 runtime
libs via its `cuda-toolkit` dependency -- changing torch would fight that.

The model uses the symbol exactly once, in `MolformerModel.forward`, to build
the 4D additive attention mask of a bidirectional encoder. This module injects
an equivalent implementation so the model's own file stays untouched.

Usage (must run BEFORE AutoModel.from_pretrained)::

    import molformer_compat
    molformer_compat.install()
"""
import torch

__all__ = ["create_bidirectional_mask", "install"]

_INSTALLED = False


def create_bidirectional_mask(config=None, inputs_embeds=None, attention_mask=None,
                              dtype=None, **kwargs):
    """Additive 4D mask for a bidirectional encoder, shape (B, 1, 1, KV).

    Allowed positions get 0.0, padding gets finfo(dtype).min, so the encoder can
    simply add it to the attention logits -- identical to what
    `BertModel.get_extended_attention_mask` produced in transformers 4.x.
    """
    if inputs_embeds is None and attention_mask is None:
        raise ValueError("create_bidirectional_mask needs inputs_embeds or attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
    target_dtype = dtype if dtype is not None else inputs_embeds.dtype
    extended = attention_mask[:, None, None, :].to(dtype=target_dtype)
    return (1.0 - extended) * torch.finfo(target_dtype).min


def install(verbose=True):
    """Inject the shim into transformers.masking_utils if it is missing."""
    global _INSTALLED
    if _INSTALLED:
        return True
    import transformers
    from transformers import masking_utils

    if hasattr(masking_utils, "create_bidirectional_mask"):
        _INSTALLED = True
        if verbose:
            print("[molformer_compat] transformers %s already provides "
                  "create_bidirectional_mask, nothing to do" % transformers.__version__,
                  flush=True)
        return True

    masking_utils.create_bidirectional_mask = create_bidirectional_mask
    _INSTALLED = True
    if verbose:
        print("[molformer_compat] injected create_bidirectional_mask into "
              "transformers %s (shim active)" % transformers.__version__, flush=True)
    return True
