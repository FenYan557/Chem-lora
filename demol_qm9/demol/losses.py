"""Pre-training / fine-tuning objectives and label standardisation.

Paper Appendix D.1 defines four pre-training losses:

``Lprop``   MAE on the quantum property (HOMO-LUMO gap for PCQM4Mv2).  For the
            QM9 protocol (Appendix D.4) training uses MSE and evaluation uses
            MAE, with label standardisation, so ``prop_mode`` switches between
            the two.
``Lmask``   cross-entropy on the masked atom types.
``Lcoord``  MSE between the recovered and the ground-truth coordinates.
``Lbond``   BCE over atom pairs, positives given by Algorithm 1 (covalent radii,
            alpha = 1.15).  Only pairs inside the structure-aware mask are
            scored, which is exactly the paper's ``E_all`` for this model: the
            mask keeps every pair closer than 5 A, so all true bonds are
            included while the O(N^2) far field is dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qm9 import N_TARGETS

__all__ = ["LossConfig", "DeMolLoss", "TargetStandardizer", "mae_per_target"]


# --------------------------------------------------------------------------
# label standardisation (Appendix D.4: "we also adopt label standardisation")
# --------------------------------------------------------------------------
class TargetStandardizer:
    """Per-target mean/std fitted on the training split."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetStandardizer":
        targets = np.asarray(targets, dtype=np.float64)
        return cls(targets.mean(0), targets.std(0))

    def normalize(self, x):
        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return (x - mean) / std
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x):
        if torch.is_tensor(x):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return x * std + mean
        return (np.asarray(x, dtype=np.float64) * self.std + self.mean).astype(np.float64)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def load(cls, path: str) -> "TargetStandardizer":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(np.asarray(data["mean"]), np.asarray(data["std"]))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.state_dict(), fh, indent=2)


# --------------------------------------------------------------------------
# loss configuration
# --------------------------------------------------------------------------
@dataclass
class LossConfig:
    w_prop: float = 1.0
    w_mask: float = 1.0
    w_coord: float = 1.0
    w_bond: float = 1.0
    prop_mode: str = "mse"            # "mse" (QM9 fine-tuning) | "mae" (paper pre-training)
    target_indices: tuple[int, ...] = tuple(range(N_TARGETS))
    target_weights: tuple[float, ...] | None = None   # None -> uniform
    bond_pos_weight: float = 0.0      # 0 -> estimated from the batch
    coord_mode: str = "delta"         # must match DeMolConfig.coord_target

    def __post_init__(self) -> None:
        if self.prop_mode not in ("mse", "mae"):
            raise ValueError("prop_mode must be 'mse' or 'mae'")
        if self.target_weights is None:
            self.target_weights = tuple(1.0 for _ in self.target_indices)
        if len(self.target_weights) != len(self.target_indices):
            raise ValueError("target_weights and target_indices must have equal length")


def mae_per_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, T) predictions/labels -> (T,) mean absolute error."""
    return (pred - target).abs().mean(dim=0)


# --------------------------------------------------------------------------
# combined objective
# --------------------------------------------------------------------------
class DeMolLoss(nn.Module):
    """Weighted sum of the four DeMol objectives; returns (total, components)."""

    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()

    # -- individual terms -------------------------------------------------
    def property_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        idx = torch.as_tensor(cfg.target_indices, device=pred.device, dtype=torch.long)
        p = pred.index_select(-1, idx)
        t = target.index_select(-1, idx)
        w = torch.as_tensor(cfg.target_weights, dtype=p.dtype, device=p.device)
        if cfg.prop_mode == "mae":
            per = (p - t).abs().mean(dim=0)
        else:
            per = ((p - t) ** 2).mean(dim=0)
        return (per * w).sum() / w.sum().clamp_min(1e-8)

    def mask_loss(
        self, atom_logits: torch.Tensor, atom_class: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor:
        if not bool(masked.any()):
            return atom_logits.sum() * 0.0
        logits = atom_logits[masked]
        labels = atom_class[masked]
        return F.cross_entropy(logits, labels)

    def coord_loss(
        self,
        coord_pred: torch.Tensor,
        coord_target: torch.Tensor,
        coord_noise: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.coord_mode == "noise":
            target = coord_noise
        else:
            target = coord_target
        sq = ((coord_pred - target) ** 2).sum(-1)                  # (B, N)
        denom = atom_mask.sum().clamp_min(1).to(sq.dtype)
        return (sq * atom_mask.to(sq.dtype)).sum() / denom

    def bond_loss(
        self, bond_logits: torch.Tensor, bond_gt: torch.Tensor, aa_mask: torch.Tensor
    ) -> torch.Tensor:
        """BCE over the upper triangle of the structure-aware atom-pair mask."""
        n = bond_logits.shape[-1]
        triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=bond_logits.device), diagonal=1)
        valid = aa_mask & triu[None]
        if not bool(valid.any()):
            return bond_logits.sum() * 0.0
        logits = bond_logits[valid]
        labels = bond_gt[valid].to(logits.dtype)
        pos_weight = None
        if self.cfg.bond_pos_weight > 0:
            pos_weight = torch.as_tensor(
                self.cfg.bond_pos_weight, dtype=logits.dtype, device=logits.device
            )
        else:
            n_pos = labels.sum()
            n_neg = labels.numel() - n_pos
            if n_pos > 0:
                pos_weight = (n_neg / n_pos).clamp(1.0, 50.0).detach()
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    # -- combined ---------------------------------------------------------
    def forward(
        self, out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """``out`` is the :class:`~demol.model.DeMol` output, ``batch`` the
        collated input (it carries ``targets``, ``atom_class`` and ``bond_gt``).
        """
        cfg = self.cfg
        targets = batch["targets"]
        total = targets.new_zeros(())
        parts: dict[str, float] = {}

        if cfg.w_prop != 0.0:
            l_prop = self.property_loss(out["prop"], targets)
            total = total + cfg.w_prop * l_prop
            parts["prop"] = float(l_prop.detach())
        if cfg.w_mask != 0.0:
            l_mask = self.mask_loss(
                out["atom_logits"], batch["atom_class"].long(), out["masked_atoms"]
            )
            total = total + cfg.w_mask * l_mask
            parts["mask"] = float(l_mask.detach())
        if cfg.w_coord != 0.0:
            l_coord = self.coord_loss(
                out["coord_pred"], out["coord_target"], out["coord_noise"], out["atom_mask"]
            )
            total = total + cfg.w_coord * l_coord
            parts["coord"] = float(l_coord.detach())
        if cfg.w_bond != 0.0:
            l_bond = self.bond_loss(
                out["bond_logits"], batch["bond_gt"].bool(), out["aa_mask"]
            )
            total = total + cfg.w_bond * l_bond
            parts["bond"] = float(l_bond.detach())

        parts["total"] = float(total.detach())
        return total, parts
