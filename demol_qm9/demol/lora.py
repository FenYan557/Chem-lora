"""LoRA adapters for DeMol fine-tuning (Hu et al., 2021, arXiv:2106.09685).

研究问题（chem-lora）：QM9 预训练的 DeMol 在 PCQM4Mv2 上用 LoRA 微调，
能追到 PCQ 全量预训练结果的多少。LoRA 冻结骨干、只训练低秩增量，
正好隔离"表示可迁移性"和"容量"两个因素。

用法::

    model = DeMol(cfg)
    load_pretrained_weights(model, ckpt)          # 先载预训练权重
    n = apply_lora(model, rank=8, alpha=16)       # 再注入 adapter
    mark_lora_trainable(model, train_embeddings=True)
    trainer = Trainer(model, ...)                 # 最后建优化器

顺序不能换：先注入 LoRA 再载权重，权重加载不会动 adapter（B 保持零初始化，
等价于无 adapter），但为了日志里"可训练参数量"准确，还是先注入再建 Trainer。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "LoRALinear",
    "apply_lora",
    "lora_parameters",
    "mark_lora_trainable",
    "adapter_state_dict",
    "load_adapter",
]

# 默认注入点：每个 DoubleHelixBlock 的 4 条注意力（q/k/v/o）+ 4 个 FFN（fc1/fc2）
DEFAULT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2",
)


class LoRALinear(nn.Module):
    """``nn.Linear`` + 低秩旁路：``y = W x + (alpha / rank) * B A x``。

    B 零初始化，训练起点严格等于冻结的基座；A 用 kaiming uniform（LoRA 论文
    的初始化）。``base`` 被冻结（requires_grad=False）。
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (self.lora_drop(x) @ self.lora_A.t()) @ self.lora_B_t()
        return out + delta * self.scaling

    def lora_B_t(self) -> torch.Tensor:
        return self.lora_B.t()


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES,
) -> int:
    """把所有后缀匹配的 ``nn.Linear`` 换成 :class:`LoRALinear`，返回注入数量。"""
    suffixes = tuple(target_suffixes)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in suffixes:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child, module))
    for parent, child, linear in replacements:
        setattr(parent, child, LoRALinear(linear, rank, alpha, dropout))
    return len(replacements)


def lora_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def mark_lora_trainable(
    model: nn.Module,
    train_embeddings: bool = True,
    train_heads: tuple[str, ...] = ("prop_head",),
) -> dict[str, int]:
    """冻结全模型，只放开 LoRA 参数 + 指定的头 +（可选）词嵌入。

    ``train_embeddings=True`` 的理由：QM9 预训练检查点的词表只有 6 类，
    PCQ 新增的 8 个元素（B/Si/P/S/Cl/Br/I/Se）对应的嵌入行是随机初始化的，
    不放开就学不到。旧元素的行已经在预训练里学好了，跟着微调影响很小。

    返回各类可训练参数量，用于日志核对。
    """
    for p in model.parameters():
        p.requires_grad_(False)
    groups: dict[str, int] = {"lora": 0, "heads": 0, "embeddings": 0}
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
            groups["lora"] += p.numel()
        elif any(name.startswith(h) for h in train_heads):
            p.requires_grad_(True)
            groups["heads"] += p.numel()
        elif train_embeddings and (
            name.startswith("atom_emb") or name.startswith("bond_emb")
            or name == "mask_token"
        ):
            p.requires_grad_(True)
            groups["embeddings"] += p.numel()
    groups["total"] = sum(groups.values())
    return groups


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只含 LoRA 参数（以及约定一起保存的属性头）的轻量 state dict。"""
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if "lora_" in k
    }


def load_adapter(model: nn.Module, state: dict[str, Any]) -> None:
    own = model.state_dict()
    kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(kept, strict=False)
