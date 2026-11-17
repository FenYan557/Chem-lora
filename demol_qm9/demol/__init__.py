"""DeMol (Liu et al., ICLR 2026) reproduction targeted at QM9 pre-training.

Paper: "Enhancing Molecular Property Predictions by Learning from Bond
Modelling and Interactions", arXiv:2603.00568.  No official code is released,
so this package is a from-scratch reimplementation driven by the paper text
(Section 3.2, Section 4, Appendix B/C/D) plus explicitly documented choices
wherever the paper is ambiguous.  See README.md for the paper -> code map.
"""

from .chem import BOND_THRESHOLD_ALPHA, COVALENT_RADII, perceive_bonds
from .benchmarks import PAPER_QM9_MAE, PAPER_QM9_NO_PRETRAIN, comparison_table
from .configs import loss_config, model_config, train_config
from .losses import DeMolLoss, LossConfig, TargetStandardizer
from .model import DeMol, DeMolConfig
from .trainer import EMA, TrainConfig, Trainer

__all__ = [
    "COVALENT_RADII",
    "BOND_THRESHOLD_ALPHA",
    "perceive_bonds",
    "PAPER_QM9_MAE",
    "PAPER_QM9_NO_PRETRAIN",
    "comparison_table",
    "DeMol",
    "DeMolConfig",
    "DeMolLoss",
    "LossConfig",
    "TargetStandardizer",
    "TrainConfig",
    "Trainer",
    "EMA",
    "model_config",
    "train_config",
    "loss_config",
]
