#!/usr/bin/env python
"""随机数据探针：确认 DeMol 的前向、反向、优化器三步真的连在一起。

``smoke_test.py`` 是 11 项逐条断言，这个脚本只做一件事——拿一批随机合成分子
（随机重原子树 + 按价态补氢，目标值是随机数）喂进真实的
``DeMol`` -> ``DeMolLoss`` -> ``torch.optim.AdamW`` 链路，连续优化若干步，然后把
损失曲线、梯度范数、参数位移打印出来。因为数据是随机的，模型只要接对了就应该能
把损失压下去（过拟合这一批），所以这是"能不能训"最直接的证据。

用法::

    python scripts/random_probe.py                       # 30 步，smoke 配置，CPU
    python scripts/random_probe.py --steps 120 --preset small
    python scripts/random_probe.py --device cuda --mols 32
    python scripts/random_probe.py --stage finetune --targets gap

退出码 0 = 损失始终有限、最后一步低于第一步、每步梯度非零且有限、参数确实被更新。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 与 run_common.py 同样的 UTF-8 输出保护：中文说明在任意代码页下都不会炸
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import torch  # noqa: E402

import smoke_test as st  # noqa: E402  复用合成分子生成器与临时目录 helper
from demol.configs import loss_config, model_config  # noqa: E402
from demol.losses import DeMolLoss, TargetStandardizer  # noqa: E402
from demol.model import DeMol, count_parameters  # noqa: E402
from demol.qm9 import QM9Dataset, qm9_collate  # noqa: E402


def _describe_batch(batch: dict[str, torch.Tensor]) -> str:
    n_valid = int(batch["atom_mask"].sum())
    m_valid = int(batch["bond_mask"].sum())
    per_mol = batch["atom_mask"].sum(1).cpu().numpy()
    per_bond = batch["bond_mask"].sum(1).cpu().numpy()
    return (f"batch={per_mol.shape[0]} 个分子，稠密 N={batch['atom_mask'].shape[1]}，"
            f"M={batch['bond_mask'].shape[1]}；有效原子 {n_valid}（每个分子 "
            f"{per_mol.min()}-{per_mol.max()}），有效化学键 {m_valid}"
            f"（每个分子 {per_bond.min()}-{per_bond.max()}）")


def _grad_stats(model: torch.nn.Module) -> tuple[float, int, int, bool, set[int]]:
    """返回 (全局梯度 L2 范数, 有梯度的参数数, 非零梯度的参数数, 是否全部有限, 非零参数 id 集)。"""
    total_sq, n_grad, n_nonzero, finite = 0.0, 0, 0, True
    nonzero_ids: set[int] = set()
    for p in model.parameters():
        if p.grad is None:
            continue
        n_grad += 1
        if not bool(torch.isfinite(p.grad).all()):
            finite = False
            continue
        s = float(p.grad.detach().pow(2).sum())
        total_sq += s
        if s > 0.0:
            n_nonzero += 1
            nonzero_ids.add(id(p))
    return float(np.sqrt(total_sq)), n_grad, n_nonzero, finite, nonzero_ids


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回落到 CPU")
        device = torch.device("cpu")

    cache_dir, records = st._build_temp_cache(args.mols, seed=args.seed)
    ds = None
    try:
        ds = QM9Dataset(cache_dir, split_ids=(0,))
        batch = qm9_collate([ds[i] for i in range(len(ds))])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        std = TargetStandardizer.fit(np.stack([r["targets"] for r in records[: len(ds)]]))

        cfg = model_config(args.preset)
        model = DeMol(cfg).to(device)
        criterion = DeMolLoss(loss_config(args.stage, args.targets)).to(device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98)
        )
        n_params = count_parameters(model)
        print(f"设备 {device}；preset={args.preset}；参数 {n_params / 1e6:.3f} M；"
              f"stage={args.stage} targets={args.targets}")
        print(_describe_batch(batch))
        print(f"损失配置：{criterion.cfg}")
        print(f"优化器：AdamW lr={args.lr} wd={args.weight_decay}；"
              f"mask_ratio={args.mask_ratio} coord_noise={args.coord_noise}")
        print(f"目标标准化：mean[:3]={np.round(std.mean[:3], 3).tolist()} "
              f"std[:3]={np.round(std.std[:3], 3).tolist()}")
        print("-" * 78)

        snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
        first_loss = last_loss = float("nan")
        best_loss = float("inf")
        problems: list[str] = []
        grad_norms: list[float] = []
        ever_grad: set[int] = set()
        t0 = time.time()

        for step in range(1, args.steps + 1):
            model.train()
            out = model(batch, mask_ratio=args.mask_ratio, coord_noise=args.coord_noise)
            loss, parts = criterion(out, batch)
            if not bool(torch.isfinite(loss)):
                problems.append(f"第 {step} 步损失非有限: {float(loss)}")
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm, n_grad, n_nonzero, finite, nonzero_ids = _grad_stats(model)
            ever_grad |= nonzero_ids
            if not finite:
                problems.append(f"第 {step} 步梯度含 NaN/Inf")
                break
            if gnorm <= 0.0:
                problems.append(f"第 {step} 步全局梯度范数为 0")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()

            grad_norms.append(gnorm)
            value = float(loss.detach())
            if step == 1:
                first_loss = value
            last_loss = value
            best_loss = min(best_loss, value)
            if step == 1 or step == args.steps or step % args.log_every == 0:
                detail = ", ".join(f"{k}={v:.4g}" for k, v in parts.items() if k != "total")
                print(f"step {step:>4d}  loss={value:>10.5f}  |g|={gnorm:>9.4g}  {detail}")

        elapsed = time.time() - t0

        # 参数位移分三类看：拿到过梯度的必须动；从没拿到梯度的属于"该阶段用不到"
        # （微调关掉 Lmask/Lcoord/Lbond 后，对应的头本来就不该更新）。
        moved, stalled, unused, max_delta = 0, [], [], 0.0
        for n, p in model.named_parameters():
            d = float((p.detach() - snapshot[n]).abs().max())
            max_delta = max(max_delta, d)
            if id(p) not in ever_grad:
                unused.append(n)
            elif d > 0.0:
                moved += 1
            else:
                stalled.append(n)
        if stalled:
            problems.append(f"{len(stalled)} 个拿到梯度的参数没被优化器更新，例如 {stalled[:3]}")
        if unused and args.stage == "pretrain":
            problems.append(f"预训练阶段有 {len(unused)} 个参数从未收到梯度，例如 {unused[:3]}")

        drop = (1 - last_loss / first_loss) * 100 if first_loss else float("nan")
        print("-" * 78)
        print(f"{args.steps} 步用时 {elapsed:.2f}s（{elapsed / max(args.steps, 1) * 1000:.1f} ms/步）")
        print(f"loss: 首步 {first_loss:.5f} -> 末步 {last_loss:.5f}"
              f"（最低 {best_loss:.5f}，降幅 {drop:.1f}%）")
        if grad_norms:
            print(f"梯度范数：首步 {grad_norms[0]:.4g}，末步 {grad_norms[-1]:.4g}，"
                  f"最大 {max(grad_norms):.4g}")
        n_touched = len(snapshot) - len(unused)
        print(f"参数位移：{moved}/{n_touched} 个参与梯度的参数被更新，最大变化 {max_delta:.4g}")
        if unused:
            print(f"另有 {len(unused)} 个参数该阶段用不到（未收到梯度），例如 {unused[:4]}")

        if not np.isfinite(first_loss) or not np.isfinite(last_loss):
            problems.append("损失出现非有限值")
        elif last_loss >= first_loss:
            problems.append(f"损失没有下降（{first_loss:.5f} -> {last_loss:.5f}）")

        if problems:
            print("\n结论：探针未通过")
            for p in problems:
                print(f"  FAIL {p}")
            return 1
        print("\n结论：前向、反向、优化器三步全部连通——损失有限且下降，"
              "所有参数都拿到非零有限梯度并被真正更新。")
        return 0
    finally:
        st._cleanup([cache_dir], [ds])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="随机数据上的前向/反向/优化探针")
    ap.add_argument("--steps", type=int, default=30, help="优化步数（默认 30）")
    ap.add_argument("--mols", type=int, default=8, help="随机分子数（默认 8，整批喂入）")
    ap.add_argument("--preset", default="smoke", choices=("smoke", "small", "base", "paper"))
    ap.add_argument("--stage", default="pretrain", choices=("pretrain", "finetune"))
    ap.add_argument("--targets", default="all", help="监督哪些目标，同 run_pretrain.py")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪范数（默认 1.0）")
    ap.add_argument("--mask-ratio", type=float, default=0.2, help="Lmask 的掩码比例")
    ap.add_argument("--coord-noise", type=float, default=0.1, help="Lcoord 的坐标噪声 std (A)")
    ap.add_argument("--device", default="cpu", help="cpu 或 cuda")
    ap.add_argument("--threads", type=int, default=1, help="CPU 线程数（默认 1，保证可复现）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5, help="每几步打印一次")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except Exception:  # noqa: BLE001  探针失败时保留完整栈便于排查
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
