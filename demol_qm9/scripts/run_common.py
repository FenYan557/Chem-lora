#!/usr/bin/env python
"""Shared plumbing for the DeMol run scripts (pre-training and fine-tuning).

Both entry points need the same things: resolve ``--preset`` / ``--scale`` into
configs, apply ``--set-model key=value`` style overrides, build the QM9 cache
loaders, fit the label standardiser on the training split, wrap everything in a
:class:`~demol.trainer.Trainer`, and write the artefacts next to the
checkpoints.  Keeping that here means the two scripts only describe what is
actually different between the two stages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows 控制台默认使用传统代码页，中文日志在非中文区域设置下会直接抛
# UnicodeEncodeError；stdout/stderr 支持 reconfigure 时统一切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import torch  # noqa: E402

from demol.configs import apply_overrides, loss_config, model_config, train_config  # noqa: E402
from demol.data import DataBundle, build_loaders, fit_standardizer  # noqa: E402
from demol.losses import LossConfig, TargetStandardizer  # noqa: E402
from demol.model import DeMol, DeMolConfig  # noqa: E402
from demol.qm9 import TARGET_NAMES  # noqa: E402
from demol.trainer import Trainer, TrainConfig, set_seed, summarize_model  # noqa: E402

__all__ = [
    "MODEL_PRESETS",
    "SCALES",
    "RunBundle",
    "add_common_args",
    "build_configs",
    "build_run",
    "describe_environment",
    "export_ema_checkpoint",
    "make_logger",
    "parse_kv",
    "print_param_summary",
    "report_splits",
    "resolve_device",
    "save_run_config",
    "selected_target_names",
    "write_json",
]

MODEL_PRESETS = ("smoke", "small", "base", "paper")
SCALES = ("paper", "local")


# --------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------
def parse_kv(items: Iterable[str] | None) -> dict[str, str]:
    """``["lr=1e-4", "n_layers=3"]`` -> ``{"lr": "1e-4", "n_layers": "3"}``."""
    out: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"[err ] expected key=value, got '{item}'")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def add_common_args(ap: argparse.ArgumentParser, stage: str) -> argparse.ArgumentParser:
    """Options shared by pre-training and fine-tuning."""
    ap.add_argument("--cache-dir", default=os.path.join("data", "qm9_cache"),
                    help="directory written by scripts/preprocess_qm9.py")
    ap.add_argument("--out-dir", default=os.path.join("runs", stage),
                    help="checkpoints, configs and logs go here")
    ap.add_argument("--preset", choices=MODEL_PRESETS, default="base",
                    help="model size: smoke/small/base/paper (Appendix D.1)")
    ap.add_argument("--scale", choices=SCALES, default="local",
                    help="'paper' = Appendix D optimisation budget, 'local' = one 8 GB GPU")
    ap.add_argument("--targets", default="all",
                    help="'all' or a comma-separated list, e.g. 'homo,lumo,gap'")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (falls back to CPU)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set-model", action="append", default=[], metavar="KEY=VALUE",
                    help="override a DeMolConfig field, e.g. --set-model n_layers=6")
    ap.add_argument("--set-train", action="append", default=[], metavar="KEY=VALUE",
                    help="override a TrainConfig field, e.g. --set-train total_steps=20000")
    ap.add_argument("--set-loss", action="append", default=[], metavar="KEY=VALUE",
                    help="override a LossConfig field, e.g. --set-loss w_bond=0.5")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--dry-run", action="store_true",
                    help="build configs + model, print the parameter budget and exit "
                         "without touching the dataset or running a single step")
    return ap


def resolve_device(name: str) -> str:
    """Honour ``--device`` but never fail on a machine without CUDA."""
    if name in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if name == "cuda":
            print("[warn] --device cuda but CUDA is unavailable -> using CPU", file=sys.stderr)
        return "cpu"
    return "cpu"


def describe_environment(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device == "cuda" and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def build_configs(args: argparse.Namespace, stage: str) -> tuple[DeMolConfig, TrainConfig, LossConfig]:
    """Resolve presets + CLI overrides into the three config dataclasses."""
    model_cfg = apply_overrides(model_config(args.preset), parse_kv(args.set_model))
    train_cfg = apply_overrides(
        train_config(
            stage,
            args.scale,
            out_dir=args.out_dir,
            seed=args.seed,
            num_workers=args.num_workers,
        ),
        parse_kv(args.set_train),
    )
    loss_cfg = apply_overrides(loss_config(stage, args.targets), parse_kv(args.set_loss))
    return model_cfg, train_cfg, loss_cfg


@dataclass
class RunBundle:
    """Everything a run script needs after setup."""

    model_cfg: DeMolConfig
    train_cfg: TrainConfig
    loss_cfg: LossConfig
    model: DeMol
    trainer: Trainer
    data: DataBundle
    standardizer: TargetStandardizer
    device: str
    rank: int = 0
    world_size: int = 1
    is_main: bool = True


def _require_cache(cache_dir: str) -> None:
    needed = ("index.npy", "targets.npy", "atom_offsets.npy", "bond_offsets.npy")
    missing = [n for n in needed if not os.path.exists(os.path.join(cache_dir, n))]
    if missing:
        raise SystemExit(
            f"[err ] {cache_dir} is not a QM9 cache (missing {', '.join(missing)}).\n"
            "       run:  python scripts/download_qm9.py --out data/qm9\n"
            f"       then: python scripts/preprocess_qm9.py --raw-dir data/qm9/dsgdb9nsd.xyz "
            f"--cache-dir {cache_dir}"
        )


def build_run(args: argparse.Namespace, stage: str) -> RunBundle:
    """Configs -> data -> standardiser -> model -> trainer.

    由 torchrun 启动（``torchrun --nproc_per_node=4 ...``）时自动进入 DDP：
    每个进程绑定 LOCAL_RANK 对应的 GPU，训练集按 rank 分片，日志与检查点
    只由 rank 0 写出。``train_cfg.batch_size`` 是**每卡** batch，
    全局 batch = batch_size × world_size。
    """
    model_cfg, train_cfg, loss_cfg = build_configs(args, stage)
    _require_cache(args.cache_dir)
    if os.environ.get("DEMOL_FAULT_HANDLER"):
        # 调试钩子：kill -USR1 <pid> 可向 stderr 转储所有线程栈。
        import faulthandler
        import signal

        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    if ddp:
        import torch.distributed as dist

        nccl_ok = torch.cuda.is_available() and dist.is_nccl_available()
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if nccl_ok else "gloo")
        if nccl_ok:
            torch.cuda.set_device(local_rank)
    set_seed(train_cfg.seed)
    use_cuda_ddp = False
    if ddp:
        import torch.distributed as dist

        use_cuda_ddp = torch.cuda.is_available() and dist.is_nccl_available()
    device = f"cuda:{local_rank}" if use_cuda_ddp else resolve_device(args.device)

    data = build_loaders(
        args.cache_dir,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        seed=train_cfg.seed,
        rank=rank,
        world_size=world_size,
    )
    standardizer = fit_standardizer(data.datasets["train"])
    model = DeMol(model_cfg)
    trainer = Trainer(
        model, train_cfg, loss_cfg, standardizer, device=device,
        ddp=ddp, rank=rank, local_rank=local_rank, world_size=world_size,
    )
    return RunBundle(
        model_cfg=model_cfg, train_cfg=train_cfg, loss_cfg=loss_cfg,
        model=trainer.raw_model, trainer=trainer, data=data,
        standardizer=standardizer, device=device,
        rank=rank, world_size=world_size, is_main=(rank == 0),
    )


def ddp_cleanup(world_size: int) -> None:
    """销毁进程组（单进程时是 no-op）。

    刻意不做 barrier：非主 rank 不碰任何文件，无需等待；若某个 rank 在
    训练中抛异常，finally 里的 barrier 会把 traceback 永远憋在异常传播
    之前，毁掉现场。
    """
    if world_size <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def selected_target_names(loss_cfg: LossConfig) -> tuple[str, ...]:
    """The QM9 names this run actually optimises (loss order, de-duplicated)."""
    seen: list[str] = []
    for i in loss_cfg.target_indices:
        name = TARGET_NAMES[i]
        if name not in seen:
            seen.append(name)
    return tuple(seen)


# --------------------------------------------------------------------------
# reporting / artefacts
# --------------------------------------------------------------------------
def print_param_summary(model: DeMol, header: str = "model") -> dict[str, Any]:
    summary = summarize_model(model)
    total = max(1, summary["total"])
    print(f"[{header}] {summary['total'] / 1e6:.2f} M parameters")
    for name, n in summary["by_component"].items():
        print(f"         {name:<22} {n / 1e6:8.2f} M  ({100.0 * n / total:5.1f}%)")
    return summary


def make_logger(out_dir: str, name: str = "train_log.txt", enabled: bool = True) -> tuple[Callable[[str], None], Any]:
    """Tee ``log(msg)`` to stdout and to ``out_dir/name``.

    ``enabled=False``（DDP 的非主 rank）返回一个静默日志，不产生任何输出，
    也不会去争用同一个日志文件。
    """
    if not enabled:
        class _NullHandle:
            def write(self, *a: Any, **k: Any) -> None:
                pass

            def close(self) -> None:
                pass

        return (lambda msg: None), _NullHandle()
    os.makedirs(out_dir, exist_ok=True)
    handle = open(os.path.join(out_dir, name), "a", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        handle.write(msg + "\n")
        handle.flush()

    return log, handle


def write_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_run_config(
    out_dir: str,
    model_cfg: DeMolConfig,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig,
    extra: dict[str, Any] | None = None,
) -> str:
    """Persist the *resolved* configuration so a run is reproducible later."""
    payload: dict[str, Any] = {
        "model": model_cfg.to_dict(),
        "train": train_cfg.to_dict(),
        "loss": loss_cfg.__dict__,
    }
    payload.update(extra or {})
    path = os.path.join(out_dir, "config.json")
    write_json(path, payload)
    return path


def export_ema_checkpoint(trainer: Trainer, path: str) -> bool:
    """Write the EMA weights as a plain ``{"model": state_dict}`` checkpoint.

    Fine-tuning loads a single state dict, so the EMA shadow (which only holds
    floating-point tensors) is merged over the live state dict to keep integer
    buffers such as the Gaussian-basis centres.
    """
    if trainer.ema is None:
        return False
    state = {k: v.detach().clone() for k, v in trainer.raw_model.state_dict().items()}
    state.update({
        k: v.to(state[k].dtype) for k, v in trainer.ema.shadow.items() if k in state
    })
    torch.save(
        {"model": state, "step": trainer.step, "model_config": trainer.raw_model.cfg.to_dict()},
        path,
    )
    return True


def report_splits(
    trainer: Trainer,
    splits: dict[str, Any],
    targets: Iterable[str] | None = None,
    use_ema: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Evaluate every split and log the per-target MAE in physical units."""
    names = list(targets) if targets is not None else None
    results: dict[str, Any] = {}
    for split, loader in splits.items():
        metrics = trainer.evaluate(loader, use_ema=use_ema)
        picked = metrics["mae"] if names is None else {n: metrics["mae"][n] for n in names}
        results[split] = {
            "loss": metrics["loss"],
            "coord_rmse": metrics["coord_rmse"],
            "n": metrics["n"],
            "mae": picked,
            "mae_all_targets": metrics["mae"],
        }
        pretty = "  ".join(f"{k}={v:.5g}" for k, v in picked.items())
        log(f"[{split:<5}] n={metrics['n']:<7} loss={metrics['loss']:.5g}  {pretty}")
    return results
