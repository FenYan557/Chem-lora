#!/usr/bin/env python
"""DeMol 在 QM9 上的微调入口（Appendix D.4）。

与预训练阶段的差别：

* 损失只用 ``Lprop``，训练用 **MSE**、评估用 **MAE**（跟 Transformer-M 一致）；
* ``mask_ratio = coord_noise = 0``，三个辅助头虽然还在（结构不变，权重可直接
  继承），但不参与损失；
* 峰值 lr 7e-5、batch 128、600k step、60k warmup 后线性衰减到 0、grad clip 5.0、
  weight decay 0、dropout(input/attn/hidden) = 0.0/0.1/0.0 —— 这些都在
  ``--scale paper`` 预设里；本地用 ``--scale local`` 缩短 schedule。

属性头保持论文默认的 12 路输出，只在损失和评估里挑选 ``--targets`` 指定的列。
Appendix D.4 没有说明 12 个目标是联合微调还是每个目标单独微调（它跟随的
Transformer-M 是每个目标一个模型），两种都支持：``--targets all`` 是联合，
``--targets gap`` 是单目标。想完整复现 Table 3 就跑 12 次单目标。

用法::

    # 只检查“预训练权重能不能对上”，不训练
    python scripts/run_finetune.py --pretrained runs/pretrain_small/ema_model.pt ^
        --preset small --targets gap --dry-run

    # 单目标微调（推荐，和论文口径一致）
    python scripts/run_finetune.py --pretrained runs/pretrain_small/ema_model.pt ^
        --preset small --scale local --targets gap --out-dir runs/ft_gap

    # 不带预训练的对照组（对应论文 Table 8 的 No Pre-training）
    python scripts/run_finetune.py --preset small --scale local --targets gap ^
        --out-dir runs/ft_gap_nopre

    # 复现 Table 3 的 12 次单目标运行（PowerShell）
    foreach ($t in "mu","alpha","homo","lumo","gap","r2","zpve","u0","u298","h298","g298","cv") { ^
        python scripts/run_finetune.py --pretrained runs/pretrain/ema_model.pt ^
            --targets $t --out-dir "runs/ft_$t" }
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from demol.benchmarks import comparison_table  # noqa: E402
from demol.losses import TargetStandardizer  # noqa: E402
from demol.model import DeMol  # noqa: E402
from demol.data import cycling_loader  # noqa: E402
from run_common import (  # noqa: E402
    add_common_args,
    build_configs,
    build_run,
    ddp_cleanup,
    describe_environment,
    export_ema_checkpoint,
    make_logger,
    print_param_summary,
    report_splits,
    save_run_config,
    selected_target_names,
    write_json,
)


# --------------------------------------------------------------------------
# 权重迁移
# --------------------------------------------------------------------------
def _load_payload(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {"model": payload}
    return payload


def load_pretrained_weights(
    model: DeMol, path: str, use_ema: bool = False
) -> dict[str, Any]:
    """把预训练编码器权重装进当前模型，按形状过滤，绝不因为头不匹配而崩。

    ``strict=False`` 仍然会在**形状不一致**时报错，而微调时属性头、乃至
    ``n_targets`` 都可能与预训练不同，所以这里先按名字+形状筛一遍再加载，
    并把筛掉/缺失的键如实汇报出来。

    词表扩展（6 类 -> 14 类）造成的"前缀兼容"张量——嵌入表、Gaussian
    basis 的逐类别 alpha/beta 表——做前缀复制：旧类别索引语义不变
    （append-only），新增行保持随机初始化。每个维度都要求检查点尺寸
    <= 模型尺寸才允许前缀复制，否则跳过。
    """
    payload = _load_payload(path)
    weights = dict(payload.get("model", payload))
    if use_ema and isinstance(payload.get("ema"), dict):
        weights.update(payload["ema"])          # EMA 覆盖同名浮点权重

    own = model.state_dict()
    kept: dict[str, torch.Tensor] = {}
    prefix_copied: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    for key, value in weights.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            kept[key] = value
        elif (
            key in own
            and own[key].ndim == value.ndim
            and all(v <= o for v, o in zip(value.shape, own[key].shape))
        ):
            # 前缀复制：老权重填进新张量的左上角，其余保持初始化值
            new_value = own[key].clone()
            slices = tuple(slice(0, s) for s in value.shape)
            new_value[slices] = value
            kept[key] = new_value
            prefix_copied.append({
                "key": key,
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(own[key].shape),
            })
        else:
            mismatched.append({
                "key": key,
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(own[key].shape) if key in own else None,
            })
    result = model.load_state_dict(kept, strict=False)
    return {
        "loaded": len(kept),
        "total_in_checkpoint": len(weights),
        "prefix_copied": prefix_copied,
        "skipped_shape_mismatch": mismatched,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "used_ema": bool(use_ema and isinstance(payload.get("ema"), dict)),
        "step": payload.get("step"),
        "pretrained_model_config": payload.get("model_config"),
    }


def reinit_prop_head(model: DeMol) -> None:
    """随机重置属性头（论文没说微调时是否继承，这里给一个开关）。"""
    for module in model.prop_head:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _summarize_transfer(info: dict[str, Any], log) -> None:
    log(f"[init] 载入预训练权重: {info['loaded']}/{info['total_in_checkpoint']} 个张量"
        + ("（EMA）" if info["used_ema"] else ""))
    if info.get("prefix_copied"):
        log(f"[init] 前缀复制（词表扩展）{len(info['prefix_copied'])} 个: "
            + ", ".join(f"{d['key']} {d['checkpoint_shape']}->{d['model_shape']}"
                        for d in info["prefix_copied"][:6]))
    if info["skipped_shape_mismatch"]:
        log(f"[init] 形状不匹配被跳过 {len(info['skipped_shape_mismatch'])} 个: "
            + ", ".join(d["key"] for d in info["skipped_shape_mismatch"][:6]))
    if info["missing"]:
        log(f"[init] 模型中未被覆盖的参数 {len(info['missing'])} 个: "
            + ", ".join(info["missing"][:6]))


# --------------------------------------------------------------------------
# dry run：只做配置 + 权重对齐检查，不训练
# --------------------------------------------------------------------------
def _dry_run(args: argparse.Namespace) -> int:
    model_cfg, train_cfg, loss_cfg = build_configs(args, "finetune")
    model = DeMol(model_cfg)
    print(f"[info] preset={args.preset} scale={args.scale} targets={args.targets}")
    print(f"[info] 微调损失: prop_mode={loss_cfg.prop_mode} w_prop={loss_cfg.w_prop} "
          f"w_mask={loss_cfg.w_mask} w_coord={loss_cfg.w_coord} w_bond={loss_cfg.w_bond}")
    print(f"[info] schedule: {train_cfg.total_steps} steps, warmup {train_cfg.warmup_steps}, "
          f"lr {train_cfg.lr:g}, batch {train_cfg.batch_size}, "
          f"mask_ratio={train_cfg.mask_ratio}, coord_noise={train_cfg.coord_noise}")
    print_param_summary(model, "finetune")
    extra: dict[str, Any] = {"stage": "finetune", "dry_run": True}
    if args.pretrained:
        info = load_pretrained_weights(model, args.pretrained, use_ema=args.pretrained_ema)
        _summarize_transfer(info, print)
        extra["pretrained"] = os.path.abspath(args.pretrained)
        extra["weight_transfer"] = {k: v for k, v in info.items() if k != "pretrained_model_config"}
    else:
        print("[init] 未提供 --pretrained：从零微调（对应论文 Table 8 的 No Pre-training）")
    save_run_config(args.out_dir, model_cfg, train_cfg, loss_cfg, extra=extra)
    print(f"[ok  ] 配置已写入 {os.path.join(args.out_dir, 'config.json')}")
    print("[dry ] 未加载数据、未执行任何前向/反向")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, "finetune")
    ap.add_argument("--pretrained", default=None,
                    help="预训练检查点：ema_model.pt / last.pt / best.pt 都可以")
    ap.add_argument("--pretrained-ema", action="store_true",
                    help="当 --pretrained 是 Trainer 检查点时，取其中的 EMA 权重")
    ap.add_argument("--reinit-head", action="store_true",
                    help="载入编码器后随机重置属性头")
    ap.add_argument("--standardizer", default=None,
                    help="复用预训练阶段的 standardizer.json（默认在训练集上重新拟合）")
    ap.add_argument("--no-ema-eval", dest="use_ema", action="store_false", default=True,
                    help="评估时用实时权重而不是 EMA 权重")
    ap.add_argument("--no-test", action="store_true", help="只评估验证集")
    ap.add_argument("--no-compare", action="store_true", help="不打印与论文 Table 3 的对比表")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run(args)

    run = build_run(args, "finetune")
    log, handle = make_logger(run.train_cfg.out_dir, enabled=run.is_main)
    try:
        env = describe_environment(run.device)
        targets = selected_target_names(run.loss_cfg)
        log(f"[info] DeMol 微调 (QM9)  preset={args.preset} scale={args.scale} "
            f"targets={','.join(targets)}")
        log(f"[info] env: {env}")
        if run.world_size > 1:
            log(f"[info] DDP: world_size={run.world_size}, "
                f"per-gpu batch={run.train_cfg.batch_size}, "
                f"global batch={run.train_cfg.batch_size * run.world_size}")
        log(f"[info] split sizes: train={len(run.data.datasets['train'])} "
            f"val={len(run.data.datasets['val'])} test={len(run.data.datasets['test'])}")
        if run.is_main:
            print_param_summary(run.model, "finetune")

        if args.standardizer:
            run.trainer.standardizer = TargetStandardizer.load(args.standardizer)
            log(f"[init] 复用标签标准化统计量: {args.standardizer}")
        elif run.is_main:
            run.standardizer.save(os.path.join(run.train_cfg.out_dir, "standardizer.json"))

        transfer: dict[str, Any] | None = None
        if args.pretrained:
            transfer = load_pretrained_weights(
                run.model, args.pretrained, use_ema=args.pretrained_ema
            )
            _summarize_transfer(transfer, log)
            if args.reinit_head:
                reinit_prop_head(run.model)
                log("[init] 属性头已随机重置")
        else:
            log("[init] 未提供 --pretrained：从零微调（论文 Table 8 的 No Pre-training 对照）")

        if run.is_main:
            save_run_config(
                run.train_cfg.out_dir, run.model_cfg, run.train_cfg, run.loss_cfg,
                extra={
                    "stage": "finetune",
                    "environment": env,
                    "cache_dir": os.path.abspath(args.cache_dir),
                    "targets": list(targets),
                    "pretrained": os.path.abspath(args.pretrained) if args.pretrained else None,
                    "weight_transfer": (
                        {k: v for k, v in transfer.items() if k != "pretrained_model_config"}
                        if transfer else None
                    ),
                    "reinit_head": args.reinit_head,
                    "ema_eval": args.use_ema,
                },
            )

        if args.resume:
            payload = run.trainer.load_checkpoint(args.resume)
            log(f"[info] 从 {args.resume} 恢复，step={payload.get('step')}")

        result = run.trainer.fit(cycling_loader(run.data.train), run.data.val, log_fn=log)

        splits: dict[str, Any] = {"val": run.data.val}
        if not args.no_test:
            splits["test"] = run.data.test
        metrics = report_splits(run.trainer, splits, targets=targets,
                                use_ema=args.use_ema, log=log)

        reference_split = "test" if "test" in metrics else "val"
        measured = metrics[reference_split]["mae"]
        if not args.no_compare:
            log("")
            log(f"[cmp ] 与论文 Table 3 对比（{reference_split} 集，"
                f"{'有' if args.pretrained else '无'}预训练；本复现用 QM9 预训练，"
                "论文用 PCQM4Mv2，数值不可直接等同）")
            for line in comparison_table(
                measured, targets=targets,
                measured_label="this run", with_no_pretrain=True,
            ).splitlines():
                log("       " + line)

        if run.is_main:
            ema_path = os.path.join(run.train_cfg.out_dir, "ema_model.pt")
            export_ema_checkpoint(run.trainer, ema_path)
            write_json(
                os.path.join(run.train_cfg.out_dir, "final_metrics.json"),
                {"steps": result["steps"], "targets": list(targets), "splits": metrics,
                 "reference_split": reference_split, "measured_mae": measured,
                 "pretrained": args.pretrained, "history": result["history"][-20:]},
            )
            log(f"[ok  ] 结果已写入 {run.train_cfg.out_dir}")
    finally:
        ddp_cleanup(run.world_size)
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
