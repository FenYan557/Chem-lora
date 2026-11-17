"""Training loop for DeMol.

Implements the optimiser / schedule details of Appendix D.1 (pre-training) and
Appendix D.4 (QM9 fine-tuning): AdamW with eps 1e-8 and betas (0.9, 0.999),
gradient clipping at 5.0, linear warmup followed by linear decay to zero, EMA
with decay 0.999, weight decay 0.0 and label standardisation.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .losses import DeMolLoss, LossConfig, TargetStandardizer
from .model import DeMol, DeMolConfig, count_parameters

__all__ = ["TrainConfig", "EMA", "build_scheduler", "Trainer", "set_seed"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    # optimiser (paper: AdamW, eps 1e-8, betas (0.9, 0.999), wd 0.0)
    lr: float = 7e-5                  # 2e-4 pre-training / 7e-5 QM9 fine-tuning
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 5.0

    # schedule
    total_steps: int = 600_000
    warmup_steps: int = 60_000
    min_lr_ratio: float = 0.0         # paper decays linearly to zero

    # data
    batch_size: int = 128
    num_workers: int = 0

    # auxiliary pre-training tasks (set to 0 for plain fine-tuning)
    mask_ratio: float = 0.15
    coord_noise: float = 0.1

    # misc
    ema_decay: float = 0.999
    amp: str = "none"                 # "none" | "bf16" | "fp16"
    seed: int = 42
    out_dir: str = "runs/demol"
    log_every: int = 100
    eval_every: int = 2000
    save_every: int = 10_000

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["betas"] = list(self.betas)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        if "betas" in kwargs:
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    """Linear warmup then linear decay to ``min_lr_ratio * lr``."""

    def lr_lambda(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * (1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential moving average of the model weights (decay 0.999)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone().float() for k, v in state.items()}

    def copy_to(self, model: nn.Module) -> None:
        """Install EMA weights into ``model`` (call :meth:`restore` afterwards)."""
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }
        current = model.state_dict()
        current.update({k: v.to(current[k].dtype) for k, v in self.shadow.items() if k in current})
        model.load_state_dict(current)

    def restore(self, model: nn.Module) -> None:
        if not self.backup:
            return
        current = model.state_dict()
        current.update(self.backup)
        model.load_state_dict(current)
        self.backup = {}


class Trainer:
    """Owns model, optimiser, schedule, EMA, standardisation and checkpoints."""

    def __init__(
        self,
        model: DeMol,
        train_cfg: TrainConfig,
        loss_cfg: LossConfig | None = None,
        standardizer: TargetStandardizer | None = None,
        device: str | torch.device = "cuda",
        ddp: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.cfg = train_cfg
        self.loss_cfg = loss_cfg or LossConfig()
        self.ddp = ddp and world_size > 1
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = rank == 0
        if self.ddp and str(device).startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
        elif self.ddp:
            self.device = torch.device("cpu")   # gloo 后端的 CPU DDP（联调用）
        else:
            self.device = torch.device(
                device if torch.cuda.is_available() or str(device) == "cpu" else "cpu"
            )
        self.raw_model = model.to(self.device)
        if self.ddp:
            # 四个任务头每步都参与损失，不存在无梯度参数，关掉 find_unused 省一次遍历。
            from torch.nn.parallel import DistributedDataParallel

            if self.device.type == "cuda":
                self.model = DistributedDataParallel(
                    self.raw_model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,
                )
            else:
                self.model = DistributedDataParallel(
                    self.raw_model, find_unused_parameters=False
                )
        else:
            self.model = self.raw_model
        self.criterion = DeMolLoss(self.loss_cfg)
        self.standardizer = standardizer
        self.no_decay = ("bias", "ln_", "layer_norm", "norm", "emb")
        self.optimizer = self._build_optimizer()
        self.scheduler = build_scheduler(self.optimizer, train_cfg)
        # EMA 与检查点始终作用于 *裸* 模型：DDP 包装会给 state_dict 的 key
        # 加上 "module." 前缀，直接存会破坏单卡恢复和微调加载。
        self.ema = EMA(self.raw_model, train_cfg.ema_decay) if train_cfg.ema_decay > 0 else None
        self.step = 0
        self.best_metric = math.inf
        self.amp_dtype = {
            "none": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[train_cfg.amp]
        # loss scaling only makes sense for fp16 on a CUDA device
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=train_cfg.amp == "fp16" and self.device.type == "cuda"
        )
        os.makedirs(train_cfg.out_dir, exist_ok=True)

    # -- setup ------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or any(tok in name for tok in self.no_decay):
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=self.cfg.lr,
            betas=tuple(self.cfg.betas),
            eps=self.cfg.eps,
        )

    def to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.standardizer is None:
            return targets
        return self.standardizer.normalize(targets)

    # -- steps ------------------------------------------------------------
    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        batch = self.to_device(batch)
        batch = {**batch, "targets": self.normalize_targets(batch["targets"])}
        with torch.autocast(
            self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            out = self.model(
                batch,
                mask_ratio=self.cfg.mask_ratio,
                coord_noise=self.cfg.coord_noise,
            )
            loss, parts = self.criterion(out, batch)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.raw_model)
        self.step += 1
        parts["lr"] = self.scheduler.get_last_lr()[0]
        return parts

    @torch.no_grad()
    def evaluate(
        self, loader: Iterable[dict[str, torch.Tensor]], use_ema: bool = True
    ) -> dict[str, Any]:
        """Validation in physical units: MAE per target plus the training loss."""
        swap = use_ema and self.ema is not None
        if swap:
            self.ema.copy_to(self.raw_model)
        try:
            self.model.eval()
            loss_sum, loss_n = 0.0, 0
            coord_sq, coord_n = 0.0, 0
            preds: list[np.ndarray] = []
            gts: list[np.ndarray] = []
            for batch in loader:
                batch = self.to_device(batch)
                norm = self.normalize_targets(batch["targets"])
                out = self.model({**batch, "targets": norm}, mask_ratio=0.0, coord_noise=0.0)
                loss, _ = self.criterion(out, {**batch, "targets": norm})
                n_b = batch["targets"].shape[0]
                loss_sum += float(loss) * n_b
                loss_n += n_b
                pred = out["prop"].float()
                if self.standardizer is not None:
                    pred = self.standardizer.denormalize(pred)
                preds.append(pred.cpu().numpy())
                gts.append(batch["targets"].float().cpu().numpy())
                mask = out["atom_mask"]
                diff = (out["coord_pred"] - out["coord_target"]).pow(2).sum(-1)
                coord_sq += float((diff * mask).sum())
                coord_n += int(mask.sum())
        finally:
            if swap:
                self.ema.restore(self.raw_model)

        pred = np.concatenate(preds, axis=0)
        gt = np.concatenate(gts, axis=0)
        mae = np.abs(pred - gt).mean(axis=0)
        names = self._target_names(gt.shape[1])
        return {
            "loss": loss_sum / max(1, loss_n),
            "mae": {name: float(mae[i]) for i, name in enumerate(names)},
            "mae_vector": mae.tolist(),
            "coord_rmse": math.sqrt(coord_sq / max(1, coord_n)),
            "n": int(gt.shape[0]),
        }

    @staticmethod
    def _target_names(n: int) -> list[str]:
        from .qm9 import TARGET_NAMES

        return list(TARGET_NAMES[:n])

    # -- loops ------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterable[dict[str, torch.Tensor]],
        val_loader: Iterable[dict[str, torch.Tensor]] | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Run the training loop until ``cfg.total_steps`` is reached.

        ``train_loader`` must yield collated batches indefinitely (use
        :func:`cycling_loader` in the run scripts); a finite loader simply ends
        the run early, which is what the smoke test relies on.
        """
        log = log_fn or (lambda msg: None)
        if not self.is_main:
            # 非主 rank 静默跑同样的步数（DDP 集合通信要求步数对齐），
            # 日志与检查点只由 rank 0 产出。
            log = lambda msg: None
        history: list[dict[str, Any]] = []
        t0 = time.time()
        for batch in train_loader:
            parts = self.train_step(batch)
            if self.step % self.cfg.log_every == 0:
                msg = " ".join(f"{k}={v:.5g}" for k, v in parts.items())
                log(f"[step {self.step}] {msg}")
            if val_loader is not None and self.step % self.cfg.eval_every == 0:
                metrics = self.evaluate(val_loader)
                history.append({"step": self.step, **metrics})
                log(
                    f"[step {self.step}] val loss={metrics['loss']:.5g} "
                    f"coord_rmse={metrics['coord_rmse']:.5g}"
                )
                self.maybe_save_best(metrics)
            if self.step % self.cfg.save_every == 0:
                self.save_checkpoint(os.path.join(self.cfg.out_dir, "last.pt"))
            if self.step >= self.cfg.total_steps:
                break
        self.save_checkpoint(os.path.join(self.cfg.out_dir, "last.pt"))
        log(f"training finished in {(time.time() - t0) / 60:.1f} min ({self.step} steps)")
        return {"history": history, "steps": self.step}

    def maybe_save_best(self, metrics: dict[str, Any]) -> None:
        score = metrics["loss"]
        if score < self.best_metric:
            self.best_metric = score
            self.save_checkpoint(os.path.join(self.cfg.out_dir, "best.pt"), metrics=metrics)

    # -- checkpoints ------------------------------------------------------
    def save_checkpoint(self, path: str, metrics: dict[str, Any] | None = None) -> None:
        if not self.is_main:
            return
        payload = {
            "step": self.step,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_config": self.cfg.to_dict(),
            "loss_config": asdict(self.loss_cfg),
            "model_config": self.model.cfg.to_dict(),
            "best_metric": self.best_metric,
            "metrics": metrics,
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        if self.standardizer is not None:
            payload["standardizer"] = self.standardizer.state_dict()
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path: str, restore_optimizer: bool = True) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        self.raw_model.load_state_dict(payload["model"])
        if restore_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        if self.ema is not None and "ema" in payload:
            self.ema.load_state_dict(payload["ema"])
        self.step = int(payload.get("step", 0))
        self.best_metric = float(payload.get("best_metric", math.inf))
        if self.standardizer is None and "standardizer" in payload:
            self.standardizer = TargetStandardizer(
                np.asarray(payload["standardizer"]["mean"]),
                np.asarray(payload["standardizer"]["std"]),
            )
        return payload


def summarize_model(model: DeMol) -> dict[str, Any]:
    """Parameter budget per component, useful when comparing with the paper."""
    groups: dict[str, int] = {}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + param.numel()
    return {
        "total": count_parameters(model),
        "by_component": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
    }
