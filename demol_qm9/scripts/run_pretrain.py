#!/usr/bin/env python
"""DeMol 预训练入口（QM9 多任务版本）。

论文里的预训练是在 PCQM4Mv2（3.37M 分子，8x A6000，1.5M step）上做的，本地
复现不具备这个条件，所以这里按用户的要求改成 **用 QM9 自身做多任务预训练**：
四个自监督/监督目标同时训练（Appendix D.1）::

    Lprop   12 个 QM9 目标的 MAE（标准化后的标签）
    Lmask   被 mask 掉的原子类型的交叉熵        （mask_ratio = 0.15）
    Lcoord  加噪坐标 -> 原始坐标的 MSE 去噪      （噪声 std = 0.1 A）
    Lbond   Algorithm 1 判定的化学键的 BCE

预训练产物（都写在 ``--out-dir`` 下）::

    config.json         解析后的 model/train/loss 配置 + 环境信息
    standardizer.json   训练集标签均值/方差（微调阶段必须复用同一份）
    last.pt / best.pt   Trainer 检查点（含 optimizer/scheduler/EMA）
    ema_model.pt        纯 EMA 权重，直接给 run_finetune.py --pretrained 用
    final_metrics.json  val/test 的物理单位 MAE
    train_log.txt       与屏幕同步的日志

用法::

    # 只看参数量和配置，不碰数据、不跑任何一步
    python scripts/run_pretrain.py --preset base --dry-run

    # 本地 8GB GPU 的完整流程（小模型 + 缩短的 schedule）
    python scripts/run_pretrain.py --preset small --scale local ^
        --cache-dir data/qm9_cache --out-dir runs/pretrain_small

    # 论文配置（需要多卡；batch 1024 / 1.5M step）
    python scripts/run_pretrain.py --preset paper --scale paper ^
        --cache-dir data/qm9_cache --out-dir runs/pretrain_paper

接着微调::

    python scripts/run_finetune.py --pretrained runs/pretrain_small/ema_model.pt ^
        --preset small --targets gap --out-dir runs/ft_gap
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demol.data import cycling_loader  # noqa: E402
from demol.model import DeMol  # noqa: E402
from demol.benchmarks import PAPER_MODEL_PARAMS_M  # noqa: E402
from run_common import (  # noqa: E402
    add_common_args,
    build_configs,
    build_run,
    describe_environment,
    export_ema_checkpoint,
    make_logger,
    print_param_summary,
    report_splits,
    save_run_config,
    selected_target_names,
    write_json,
)


def _dry_run(args: argparse.Namespace) -> int:
    """构建配置与模型、打印参数预算，然后退出（不读数据、不前向）。"""
    model_cfg, train_cfg, loss_cfg = build_configs(args, "pretrain")
    model = DeMol(model_cfg)      # CPU, only for the parameter census
    print(f"[info] preset={args.preset} scale={args.scale} targets={args.targets}")
    print(f"[info] loss weights: prop={loss_cfg.w_prop} mask={loss_cfg.w_mask} "
          f"coord={loss_cfg.w_coord} bond={loss_cfg.w_bond}  prop_mode={loss_cfg.prop_mode}")
    print(f"[info] schedule: {train_cfg.total_steps} steps, warmup {train_cfg.warmup_steps}, "
          f"lr {train_cfg.lr:g}, batch {train_cfg.batch_size}, amp={train_cfg.amp}")
    summary = print_param_summary(model, "pretrain")
    print(f"[info] 论文 Appendix D.1 的参数量为 {PAPER_MODEL_PARAMS_M:.0f} M "
          f"(12 层 / d=768 / K=128)；当前配置为 {summary['total'] / 1e6:.1f} M，"
          "差异原因见 README.md 的“参数量”一节")
    save_run_config(args.out_dir, model_cfg, train_cfg, loss_cfg,
                    extra={"stage": "pretrain", "dry_run": True})
    print(f"[ok  ] 配置已写入 {os.path.join(args.out_dir, 'config.json')}")
    print("[dry ] 未加载数据、未执行任何前向/反向")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(ap, "pretrain")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="覆盖验证间隔（步数）")
    args = ap.parse_args(argv)

    if args.dry_run:
        return _dry_run(args)

    run = build_run(args, "pretrain")
    if args.eval_every is not None:
        run.trainer.cfg.eval_every = args.eval_every

    log, handle = make_logger(run.train_cfg.out_dir)
    try:
        env = describe_environment(run.device)
        log(f"[info] DeMol 预训练 (QM9 多任务)  stage=pretrain preset={args.preset} scale={args.scale}")
        log(f"[info] env: {env}")
        log(f"[info] cache: {args.cache_dir}")
        log(f"[info] split sizes: train={len(run.data.datasets['train'])} "
            f"val={len(run.data.datasets['val'])} test={len(run.data.datasets['test'])}")
        log(f"[info] targets: {', '.join(selected_target_names(run.loss_cfg))}")
        print_param_summary(run.model, "pretrain")

        std_path = os.path.join(run.train_cfg.out_dir, "standardizer.json")
        run.standardizer.save(std_path)
        log(f"[ok  ] 标签标准化统计量 -> {std_path}")
        save_run_config(run.train_cfg.out_dir, run.model_cfg, run.train_cfg, run.loss_cfg,
                        extra={"stage": "pretrain", "environment": env,
                               "cache_dir": os.path.abspath(args.cache_dir)})

        if args.resume:
            payload = run.trainer.load_checkpoint(args.resume)
            log(f"[info] 从 {args.resume} 恢复，step={payload.get('step')}")

        result = run.trainer.fit(
            cycling_loader(run.data.train), run.data.val, log_fn=log
        )

        metrics = report_splits(
            run.trainer,
            {"val": run.data.val, "test": run.data.test},
            targets=selected_target_names(run.loss_cfg),
            log=log,
        )
        ema_path = os.path.join(run.train_cfg.out_dir, "ema_model.pt")
        if export_ema_checkpoint(run.trainer, ema_path):
            log(f"[ok  ] EMA 权重 -> {ema_path}")
        write_json(
            os.path.join(run.train_cfg.out_dir, "final_metrics.json"),
            {"steps": result["steps"], "splits": metrics,
             "history": result["history"][-20:]},
        )
        log("[next] python scripts/run_finetune.py "
            f"--pretrained {ema_path} --preset {args.preset} --targets gap")
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
