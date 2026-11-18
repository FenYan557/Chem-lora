#!/usr/bin/env python
# 用 QM9 预训练检查点零样本测 PCQ（gap MAE），作为 LoRA 微调的下界对照。
#
# 关键点：
#   1. 模型结构从检查点自带的 model_config 恢复（旧 6 类词表、n_targets=12）；
#   2. PCQ 缓存按新 14 类词表编码，经 VocabDowngradeDataset 降级回 6 类
#      （S/P/Cl 等新元素被当成 OTHER，这是零样本的固有损耗）；
#   3. 取属性头第 4 列（QM9 gap），用检查点自带 standardizer 反标准化得到
#      meV，/1000 换成 eV 后与 PCQ 标签比 MAE；
#   4. 同时报预测常数（评估集均值）的平凡基线，判断模型是否真有信号。
#
# 提醒：QM9 gap 在 0.1-0.5 eV 量级，PCQ gap 在 3-9 eV，分布完全不重叠，
# 零样本 MAE 必然很大，这个数只看相对意义。
#
# 用法（单卡即可）：
#   ~/yangsensen/venvs/demol/bin/python scripts/eval_pcq_zeroshot.py \
#       --pretrained runs/pretrain_paper/best.pt \
#       --cache-dir data/pcq_cache --split val

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from demol.losses import TargetStandardizer
from demol.model import DeMol, DeMolConfig
from demol.pcq import VocabDowngradeDataset
from demol.qm9 import QM9Dataset, TARGET_NAMES, qm9_collate
from run_common import resolve_device

GAP_INDEX = TARGET_NAMES.index("gap")       # 4


def load_qm9_checkpoint(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = payload.get("model_config")
    if cfg_dict is None:
        raise SystemExit(f"[err ] {path} 里没有 model_config，无法恢复模型结构")
    model = DeMol(DeMolConfig.from_dict(cfg_dict))
    weights = payload.get("ema", payload.get("model", payload))
    model.load_state_dict(weights, strict=False)
    std = payload.get("standardizer")
    if std is None:
        # ema_model.pt 不带 standardizer，找同目录的 standardizer.json
        sibling = os.path.join(os.path.dirname(os.path.abspath(path)),
                               "standardizer.json")
        if not os.path.exists(sibling):
            raise SystemExit(f"[err ] 找不到标签标准化统计量: {sibling}")
        with open(sibling, encoding="utf-8") as fh:
            std = json.load(fh)
    standardizer = TargetStandardizer(np.asarray(std["mean"]), np.asarray(std["std"]))
    return model, standardizer, payload


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained", required=True,
                    help="QM9 预训练检查点（best.pt / last.pt / ema_model.pt）")
    ap.add_argument("--cache-dir", default=os.path.join("data", "pcq_cache"))
    ap.add_argument("--split", choices=("train", "val"), default="val")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="只测前 N 条（快速验证用）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    model, standardizer, payload = load_qm9_checkpoint(args.pretrained)
    model.to(device).eval()
    print(f"[info] 检查点 step={payload.get('step')}, "
          f"n_targets={model.cfg.n_targets}, 词表 {model.cfg.atom_field_sizes[0]} 类")

    base = QM9Dataset(args.cache_dir, split_ids=(0 if args.split == "train" else 1,))
    dataset = VocabDowngradeDataset(base)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=qm9_collate)

    preds, gts = [], []
    seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, mask_ratio=0.0, coord_noise=0.0)
        pred_z = out["prop"][:, GAP_INDEX].float().cpu().numpy()
        pred_mev = pred_z * standardizer.std[GAP_INDEX] + standardizer.mean[GAP_INDEX]
        preds.append(pred_mev / 1000.0)                       # meV -> eV
        gts.append(batch["targets"][:, 0].float().cpu().numpy())
        seen += len(pred_mev)
        if args.limit and seen >= args.limit:
            break
    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    if args.limit:
        pred, gt = pred[: args.limit], gt[: args.limit]

    mae_ev = float(np.abs(pred - gt).mean())
    # 平凡基线：预测常数（用评估集自身均值，是最优常数预测，最宽松的基线）
    naive_mae_ev = float(np.abs(gt - gt.mean()).mean())
    print(f"[eval] split={args.split} n={len(gt)}")
    print(f"[eval] 零样本 gap MAE = {mae_ev:.4f} eV = {mae_ev * 1000:.1f} meV")
    print(f"[eval] 平凡基线（预测均值）MAE = {naive_mae_ev:.4f} eV")
    print(f"[eval] 预测分布: mean={pred.mean():.3f} std={pred.std():.3f} eV; "
          f"标签分布: mean={gt.mean():.3f} std={gt.std():.3f} eV")
    print("[eval] 解读：零样本 MAE 大于平凡基线属正常（QM9 gap ~0.2 eV vs "
          "PCQ gap ~5.7 eV，标准化输出外推不到那个量级）；若 MAE 显著小于"
          "基线说明表征里已有可迁移信号")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
