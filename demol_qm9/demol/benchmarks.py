"""Reference numbers quoted in the DeMol paper, for side-by-side reporting.

Every value here is a test-split mean absolute error on QM9 in the unit the
paper reports (Table 3 for the full benchmark, Appendix H / Table 8 for the
pre-training ablation).  They are *reporting* constants only: nothing in the
model, the loss or the trainer imports them, so a mismatch with the paper can
never silently change a run.

Two caveats that matter when reading a comparison table produced from these
numbers:

* The paper pre-trains on PCQM4Mv2 (3.37 M molecules) and then fine-tunes on
  QM9.  This reproduction pre-trains on QM9 itself, so the QM9 fine-tuning
  numbers are not expected to match -- the auxiliary-task signal comes from a
  much smaller and much less diverse corpus.
* Appendix D.4 does not say whether the 12 targets are fine-tuned with one
  joint model or one model per target (Transformer-M, which the paper follows
  for the loss, uses one model per target).  ``scripts/run_finetune.py``
  supports both; the table below is per-target.
"""

from __future__ import annotations

from .qm9 import QM9_TARGETS, TARGET_NAMES

__all__ = [
    "PAPER_QM9_MAE",
    "PAPER_QM9_NO_PRETRAIN",
    "PAPER_QM9_AVG_RANK",
    "PAPER_MODEL_PARAMS_M",
    "QM9_UNITS",
    "paper_mae",
    "comparison_table",
]

# Table 3, "DeMol" row (last model in the table).  Units: mu [D],
# alpha [Bohr^3], homo/lumo/gap/zpve [meV], r2 [Bohr^2],
# u0/u298/h298/g298 [kcal/mol], cv [cal/(mol*K)].
PAPER_QM9_MAE: dict[str, float] = {
    "mu": 0.024,
    "alpha": 0.042,
    "homo": 16.4,
    "lumo": 15.7,
    "gap": 26.8,
    "r2": 0.101,
    "zpve": 1.16,
    "u0": 6.24,
    "u298": 6.10,
    "h298": 6.35,
    "g298": 6.29,
    "cv": 0.021,
}

# Appendix H, Table 8: same architecture fine-tuned *without* the PCQM4Mv2
# pre-training stage, five targets only.
PAPER_QM9_NO_PRETRAIN: dict[str, float] = {
    "homo": 21.2,
    "lumo": 19.3,
    "gap": 30.4,
    "g298": 10.97,
    "cv": 0.029,
}

# Table 3, last column: DeMol's average rank across the 12 targets.
PAPER_QM9_AVG_RANK = 2.67

# Appendix D.1: 186 M parameters for the 12-layer / d=768 / K=128 model.
PAPER_MODEL_PARAMS_M = 186.0

QM9_UNITS: dict[str, str] = {t.name: t.unit for t in QM9_TARGETS}


def paper_mae(name: str, pretrained: bool = True) -> float | None:
    """Reported MAE for one target, or ``None`` when the paper omits it."""
    table = PAPER_QM9_MAE if pretrained else PAPER_QM9_NO_PRETRAIN
    return table.get(name)


def _fmt(value: float | None, width: int) -> str:
    if value is None:
        return "-".rjust(width)
    # keep the paper's own precision: 3 significant digits is plenty
    text = f"{value:.4g}"
    return text.rjust(width)


def comparison_table(
    measured: dict[str, float],
    targets: tuple[str, ...] | list[str] | None = None,
    measured_label: str = "this run",
    with_no_pretrain: bool = True,
) -> str:
    """Render ``measured`` (target -> MAE) against the paper's Table 3.

    ``measured`` must already be in physical units, i.e. the ``mae`` dict
    returned by :meth:`demol.trainer.Trainer.evaluate`.
    """
    names = [n for n in (targets or list(TARGET_NAMES)) if n in QM9_UNITS]
    header = (
        f"{'target':<7}{'unit':<11}{measured_label:>11}{'paper':>9}"
        f"{'ratio':>8}"
    )
    if with_no_pretrain:
        header += f"{'no-pretr.':>11}"
    lines = [header, "-" * len(header)]
    for name in names:
        ours = measured.get(name)
        paper = PAPER_QM9_MAE.get(name)
        ratio = (ours / paper) if (ours is not None and paper) else None
        line = (
            f"{name:<7}{QM9_UNITS[name]:<11}{_fmt(ours, 11)}{_fmt(paper, 9)}"
            f"{_fmt(ratio, 8)}"
        )
        if with_no_pretrain:
            line += _fmt(PAPER_QM9_NO_PRETRAIN.get(name), 11)
        lines.append(line)
    return "\n".join(lines)
