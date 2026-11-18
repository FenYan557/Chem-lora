#!/usr/bin/env python
# QM9 预训练 DeMol -> PCQM4Mv2 的 LoRA 微调入口。
#
# 研究问题：QM9 预训练的表示，用 LoRA 在 PCQ 上微调，valid MAE 能追到
# 论文 PCQ 预训练结果（Table 1，DeMol 60.3 meV）的多少。
#
# 与 QM9 微调（run_finetune.py）的差别：
#   1. 数据是 PCQ 缓存（targets 只有 gap 一列，单位 eV，元素词表 14 类）；
#   2. 模型 n_targets=1，元素词表按 features.py 扩展后的 14 类构建，
#      旧检查点的嵌入表/basis 表走前缀复制（append-only 保证语义对齐）；
#   3. 默认冻结骨干，只训练 LoRA adapter + 属性头 + 词嵌入
#      （--no-lora 可切换为全量微调对照）。
#
# 用法（A100，4 卡 DDP）：
#   ~/yangsensen/venvs/demol/bin/torchrun --nproc_per_node=4 \
#       scripts/run_pcq_finetune.py \
#       --cache-dir data/pcq_cache \
#       --pretrained runs/pretrain_paper/ema_model.pt \
#       --preset paper --out-dir runs/pcq_lora \
#       --set-train batch_size=256 --set-train total_steps=20000 \
#       --set-train amp=bf16

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from demol.benchmarks import pcq_comparison_table
from demol.configs import apply_overrides, model_config
from demol.data import cycling_loader
from demol.lora import apply_lora, lora_parameters, mark_lora_trainable
from demol.losses import LossConfig, TargetStandardizer
from demol.model import DeMol
from demol.qm9 import QM9Dataset, qm9_collate
from demol.trainer import Trainer, TrainConfig, set_seed
from run_common import (
    ddp_cleanup,
    describe_environment,
    export_ema_checkpoint,
    make_logger,
    parse_kv,
    print_param_summary,
    resolve_device,
    save_run_config,
    write_json,
)
from run_finetune import _summarize_transfer, load_pretrained_weights


def build_pcq_loaders(cache_dir, batch_size, num_workers, seed, rank, world_size):
    """train/val 两个 loader；PCQ 缓存没有 test split。"""
    ds_train = QM9Dataset(cache_dir, split_ids=(0,))
    ds_val = QM9Dataset(cache_dir, split_ids=(1,))
    common = dict(collate_fn=qm9_collate, num_workers=max(0, int(num_workers)))
    if num_workers > 0:
        common.update(persistent_workers=True, pin_memory=True)
    sampler = None
    generator = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
    train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        drop_last=True, generator=generator, **common)
    val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, **common)
    return train, val, ds_train, ds_val


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=os.path.join("data", "pcq_cache"))
    ap.add_argument("--out-dir", default=os.path.join("runs", "pcq_lora"))
    ap.add_argument("--preset", choices=("smoke", "small", "base", "paper"), default="paper")
    ap.add_argument("--pretrained", required=True, help="QM9 预训练检查点（ema_model.pt 等）")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--no-lora", action="store_true", help="全量微调对照组")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="连词嵌入也冻结（默认放开：新增元素行是随机初始化的）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set-model", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--set-train", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # ---- 配置：单目标 gap（eV），只留属性损失（MSE 训练 / MAE 评估）------
    model_cfg = apply_overrides(
        model_config(args.preset, n_targets=1), parse_kv(args.set_model))
    train_cfg = apply_overrides(TrainConfig(
        lr=5e-4,                    # LoRA 惯例 lr 比全量微调高一个量级
        batch_size=256,             # 每卡；4 卡全局 1024
        total_steps=20_000, warmup_steps=1000,
        weight_decay=0.0, grad_clip=5.0, ema_decay=0.999,
        mask_ratio=0.0, coord_noise=0.0,
        amp="bf16", seed=args.seed, out_dir=args.out_dir,
        num_workers=args.num_workers,
        log_every=50, eval_every=500, save_every=2000,
    ), parse_kv(args.set_train))
    if args.no_lora:
        train_cfg = apply_overrides(train_cfg, {"lr": "7e-5"})
    loss_cfg = LossConfig(
        w_prop=1.0, w_mask=0.0, w_coord=0.0, w_bond=0.0,
        prop_mode="mse", target_indices=(0,),
    )

    if args.dry_run:
        model = DeMol(model_cfg)
        info = load_pretrained_weights(model, args.pretrained)
        _summarize_transfer(info, print)
        if not args.no_lora:
            n = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                           dropout=args.lora_dropout)
            groups = mark_lora_trainable(
                model, train_embeddings=not args.freeze_embeddings)
            print(f"[lora] 注入 {n} 个 adapter, rank={args.lora_rank} "
                  f"alpha={args.lora_alpha}")
            print(f"[lora] 可训练参数: {groups}")
        print_param_summary(model, "pcq-finetune")
        return 0

    # ---- DDP 环境（与 run_common.build_run 同构）-------------------------
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
    device = (f"cuda:{local_rank}"
              if ddp and torch.cuda.is_available() else resolve_device(args.device))
    is_main = rank == 0

    log, handle = make_logger(train_cfg.out_dir, enabled=is_main)
    try:
        lora_desc = "off" if args.no_lora else f"r{args.lora_rank}/a{args.lora_alpha}"
        log(f"[info] DeMol PCQ LoRA 微调  preset={args.preset} lora={lora_desc}")
        log(f"[info] env: {describe_environment(device)}")
        if ddp:
            log(f"[info] DDP: world_size={world_size}, "
                f"per-gpu batch={train_cfg.batch_size}, "
                f"global batch={train_cfg.batch_size * world_size}")

        train_loader, val_loader, ds_train, ds_val = build_pcq_loaders(
            args.cache_dir, train_cfg.batch_size, args.num_workers,
            args.seed, rank, world_size)
        log(f"[info] split sizes: train={len(ds_train)} val={len(ds_val)}")

        # 标签标准化在 train split 上拟合（eV 单位，gap 均值约 5.7）
        standardizer = TargetStandardizer.fit(
            np.asarray(ds_train.arrays["targets"])[ds_train.rows])
        if is_main:
            standardizer.save(os.path.join(train_cfg.out_dir, "standardizer.json"))

        # 顺序：建模（14 类词表） -> 载预训练权重（前缀复制） -> 注入 LoRA
        #      -> 冻结 -> 建 Trainer（优化器只收可训练参数）
        model = DeMol(model_cfg)
        transfer = load_pretrained_weights(model, args.pretrained)
        _summarize_transfer(transfer, log)

        if not args.no_lora:
            n_adapters = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                                    dropout=args.lora_dropout)
            groups = mark_lora_trainable(
                model, train_embeddings=not args.freeze_embeddings)
            log(f"[lora] 注入 {n_adapters} 个 adapter (rank={args.lora_rank}, "
                f"alpha={args.lora_alpha}); 可训练参数: "
                f"lora={groups['lora'] / 1e6:.2f}M heads={groups['heads'] / 1e6:.3f}M "
                f"emb={groups['embeddings'] / 1e6:.2f}M total={groups['total'] / 1e6:.2f}M")
        else:
            log("[info] --no-lora：全量微调对照")

        if is_main:
            print_param_summary(model, "pcq-finetune")

        trainer = Trainer(
            model, train_cfg, loss_cfg, standardizer,
            target_names=("gap",), device=device,
            ddp=ddp, rank=rank, local_rank=local_rank, world_size=world_size,
        )

        if is_main:
            save_run_config(train_cfg.out_dir, model_cfg, train_cfg, loss_cfg, extra={
                "stage": "pcq_lora_finetune",
                "pretrained": os.path.abspath(args.pretrained),
                "lora": None if args.no_lora else {
                    "rank": args.lora_rank, "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "train_embeddings": not args.freeze_embeddings,
                },
                "cache_dir": os.path.abspath(args.cache_dir),
                "target": "gap (eV)",
            })

        if args.resume:
            payload = trainer.load_checkpoint(args.resume)
            log(f"[info] 从 {args.resume} 恢复，step={payload.get('step')}")

        result = trainer.fit(cycling_loader(train_loader), val_loader, log_fn=log)

        metrics = trainer.evaluate(val_loader)
        gap_mae_ev = metrics["mae"]["gap"]
        gap_mae_mev = gap_mae_ev * 1000.0
        log(f"[val ] n={metrics['n']}  gap MAE = {gap_mae_ev:.4f} eV "
            f"= {gap_mae_mev:.2f} meV")
        log("")
        log("[cmp ] 与论文 Table 1 对比（口径提醒：论文是 PCQ 预训练零样本，"
            "我们是 QM9 预训练 + LoRA 微调；且本 val 构象若是 RDKit 生成则与"
            "论文的未知构象来源又差一层）")
        for line in pcq_comparison_table(gap_mae_mev, "QM9-pre + LoRA").splitlines():
            log("       " + line)

        if is_main:
            export_ema_checkpoint(trainer, os.path.join(train_cfg.out_dir, "ema_model.pt"))
            write_json(os.path.join(train_cfg.out_dir, "final_metrics.json"), {
                "steps": result["steps"],
                "val_gap_mae_ev": gap_mae_ev,
                "val_gap_mae_mev": gap_mae_mev,
                "history": result["history"][-20:],
            })
            log(f"[ok  ] 结果已写入 {train_cfg.out_dir}")
    finally:
        ddp_cleanup(world_size)
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
