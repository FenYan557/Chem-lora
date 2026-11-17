"""Named presets for the model, the optimiser and the loss.

Two orthogonal axes:

``preset`` (model size)
    ``smoke``  2 layers / d=64  / K=16   - unit tests, seconds on CPU
    ``small``  4 layers / d=128 / K=32   - laptop GPU, end-to-end sanity
    ``base``   8 layers / d=384 / K=64   - single 24 GB GPU, real QM9 runs
    ``paper``  12 layers / d_a=d_b=768 / K=128 / 8 heads (Appendix D.1)

``scale`` (optimisation budget)
    ``paper``  the Appendix D.1 / D.4 numbers verbatim (batch 1024 pre-training,
               batch 128 fine-tuning, 1.5M / 600k steps) - needs the 8x A6000
               budget quoted in the paper.
    ``local``  same schedule shape, scaled to one 8 GB laptop GPU so the whole
               pipeline can be exercised end to end.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from typing import Any

from .losses import LossConfig
from .model import DeMolConfig
from .qm9 import N_TARGETS, TARGET_NAMES
from .trainer import TrainConfig

__all__ = [
    "MODEL_PRESETS",
    "model_config",
    "train_config",
    "loss_config",
    "apply_overrides",
]


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": dict(
        n_layers=2, d_atom=64, d_bond=64, n_heads=4, n_kernels=16, ffn_mult=2,
        kernel_width=10.0, share_structure_encoding=True, dropout_attn=0.0,
        bond_head_rank=16,
    ),
    "small": dict(
        n_layers=4, d_atom=128, d_bond=128, n_heads=8, n_kernels=32, ffn_mult=4,
        dropout_attn=0.1,
    ),
    "base": dict(
        n_layers=8, d_atom=384, d_bond=384, n_heads=8, n_kernels=64, ffn_mult=4,
        dropout_attn=0.1,
    ),
    # Appendix D.1: 12 layers, da = db = 768, 128 Gaussian kernels, 8 heads.
    # ffn_mult=1：论文 Table 1 标称 186M，用同表的 Graphormer（47.1M，公开的
    # L12/d768/mult1 模型）校准后可知论文的 FFN 隐藏维等于模型维；本实现按
    # mult=1 实测 180.7M ≈ 186M。README "参数量"一节有完整推演。
    "paper": dict(
        n_layers=12, d_atom=768, d_bond=768, n_heads=8, n_kernels=128, ffn_mult=1,
        dropout_input=0.0, dropout_attn=0.1, dropout_hidden=0.0,
    ),
}

# Appendix D.1 (pre-training on PCQM4Mv2) and D.4 (QM9 fine-tuning).
_TRAIN: dict[tuple[str, str], dict[str, Any]] = {
    ("pretrain", "paper"): dict(
        lr=2e-4, batch_size=1024, total_steps=1_500_000, warmup_steps=150_000,
        weight_decay=0.0, grad_clip=5.0, ema_decay=0.999, eps=1e-8,
        mask_ratio=0.15, coord_noise=0.1, amp="bf16",
    ),
    ("finetune", "paper"): dict(
        lr=7e-5, batch_size=128, total_steps=600_000, warmup_steps=60_000,
        weight_decay=0.0, grad_clip=5.0, ema_decay=0.999, eps=1e-8,
        mask_ratio=0.0, coord_noise=0.0, amp="bf16",
    ),
    # same schedule shape, one 8 GB laptop GPU
    ("pretrain", "local"): dict(
        lr=3e-4, batch_size=32, total_steps=100_000, warmup_steps=5_000,
        weight_decay=0.0, grad_clip=5.0, ema_decay=0.999, eps=1e-8,
        mask_ratio=0.15, coord_noise=0.1, amp="bf16",
        log_every=50, eval_every=1000, save_every=5000,
    ),
    ("finetune", "local"): dict(
        lr=2e-4, batch_size=32, total_steps=60_000, warmup_steps=3_000,
        weight_decay=0.0, grad_clip=5.0, ema_decay=0.999, eps=1e-8,
        mask_ratio=0.0, coord_noise=0.0, amp="bf16",
        log_every=50, eval_every=500, save_every=2500,
    ),
}


def model_config(preset: str = "base", **overrides: Any) -> DeMolConfig:
    if preset not in MODEL_PRESETS:
        raise KeyError(f"unknown model preset '{preset}'; choose from {sorted(MODEL_PRESETS)}")
    return DeMolConfig.from_dict({**MODEL_PRESETS[preset], **overrides})


def train_config(
    stage: str = "pretrain", scale: str = "paper", **overrides: Any
) -> TrainConfig:
    key = (stage, scale)
    if key not in _TRAIN:
        raise KeyError(f"unknown training preset {key}; choose stage/scale from {sorted(_TRAIN)}")
    return TrainConfig.from_dict({**_TRAIN[key], **overrides})


def loss_config(
    stage: str = "pretrain",
    targets: str = "all",
    **overrides: Any,
) -> LossConfig:
    """``stage='pretrain'`` enables all four objectives with MAE on the property.

    ``stage='finetune'`` follows Appendix D.4: MSE while training, MAE while
    evaluating, no auxiliary task.  ``targets`` is either ``"all"`` or a
    comma-separated list of QM9 target names (e.g. ``"gap"`` or
    ``"homo,lumo,gap"``).
    """
    idx = _resolve_targets(targets)
    if stage == "pretrain":
        base = dict(
            w_prop=1.0, w_mask=1.0, w_coord=1.0, w_bond=1.0,
            prop_mode="mae", target_indices=idx, coord_mode="delta",
        )
    elif stage == "finetune":
        base = dict(
            w_prop=1.0, w_mask=0.0, w_coord=0.0, w_bond=0.0,
            prop_mode="mse", target_indices=idx, coord_mode="delta",
        )
    else:
        raise KeyError("stage must be 'pretrain' or 'finetune'")
    return LossConfig(**{**base, **overrides})


def _resolve_targets(targets: str) -> tuple[int, ...]:
    if targets in ("all", "", None):
        return tuple(range(N_TARGETS))
    out: list[int] = []
    for name in str(targets).split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name not in TARGET_NAMES:
            raise KeyError(f"unknown QM9 target '{name}'; choose from {TARGET_NAMES}")
        out.append(TARGET_NAMES.index(name))
    if not out:
        raise ValueError("no valid target selected")
    return tuple(out)


def apply_overrides(obj: Any, overrides: dict[str, str]) -> Any:
    """Apply ``key=value`` CLI strings to a dataclass, casting to its field type.

    Works for both flavours of config in this package: the ones that carry
    ``to_dict``/``from_dict`` round-trippers (:class:`DeMolConfig`,
    :class:`TrainConfig`) and the plain ones (:class:`LossConfig`).
    """
    if not overrides:
        return obj
    fields = {f.name: f for f in dc_fields(obj)}
    parsed: dict[str, Any] = {}
    for key, raw in overrides.items():
        if key not in fields:
            raise KeyError(f"{type(obj).__name__} has no field '{key}'")
        parsed[key] = _cast(raw, fields[key].type)
    if hasattr(obj, "to_dict") and hasattr(type(obj), "from_dict"):
        return type(obj).from_dict({**obj.to_dict(), **parsed})
    # LossConfig.__post_init__ derives target_weights from target_indices, so a
    # change of indices must not be combined with the stale weight tuple.
    if "target_indices" in parsed and "target_weights" not in parsed:
        parsed["target_weights"] = None
    return dc_replace(obj, **parsed)


def _cast(raw: str, type_hint: Any) -> Any:
    hint = type_hint if isinstance(type_hint, str) else getattr(type_hint, "__name__", str(type_hint))
    low = hint.lower()
    if low.startswith("bool"):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if low.startswith("int"):
        return int(raw)
    if low.startswith("float"):
        return float(raw)
    if low.startswith("tuple") or low.startswith("list"):
        inner = "str" if "str" in low else ("float" if "float" in low else "int")
        return tuple(_cast(v.strip(), inner) for v in raw.split(",") if v.strip())
    return raw
