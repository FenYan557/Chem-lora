#!/usr/bin/env python
"""DDP 最小端到端验证（合成数据，不需要 QM9 下载/预处理）。

造一个 16 分子的合成缓存，然后用 torchrun 起 ``--nproc`` 个进程跑 4 步
预训练，检查：进程全绿退出、rank0 的检查点没有 ``module.`` 前缀、
EMA/standardizer 等产物齐全。服务器上新环境装好后跑这个确认多卡路径可用::

    python scripts/ddp_smoke.py --nproc 2
    python scripts/ddp_smoke.py --nproc 4 --set-train amp=bf16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from demol.qm9 import write_cache  # noqa: E402
from smoke_test import _cleanup, _scratch_dir, synthetic_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, default=2, help="进程数（= GPU 数）")
    ap.add_argument("--set-train", action="append", default=[], metavar="KEY=VALUE",
                    help="透传给 run_pretrain 的 TrainConfig 覆盖，如 amp=bf16")
    ap.add_argument("--keep", action="store_true", help="保留临时目录便于排查")
    args = ap.parse_args(argv)

    cache = _scratch_dir("ddp_smoke_cache_")
    out = _scratch_dir("ddp_smoke_out_")
    records = synthetic_records(16, seed=3)
    write_cache(records, cache)
    split = np.zeros(16, dtype=np.int8)
    split[12:14] = 1   # val
    split[14:] = 2     # test
    np.save(os.path.join(cache, "split.npy"), split)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        os.path.join(_HERE, "run_pretrain.py"),
        "--preset", "smoke",
        "--cache-dir", cache,
        "--out-dir", out,
        "--set-train", "total_steps=4", "--set-train", "warmup_steps=1",
        "--set-train", "eval_every=2", "--set-train", "save_every=4",
        "--set-train", "log_every=1", "--set-train", "batch_size=2",
        "--set-train", "ema_decay=0.9",
    ]
    for kv in args.set_train:
        cmd += ["--set-train", kv]
    print("+", " ".join(cmd))
    env = dict(os.environ, USE_LIBUV=os.environ.get("USE_LIBUV", "0"))
    proc = subprocess.run(cmd, cwd=_ROOT, env=env)
    if rc := proc.returncode:
        print(f"[FAIL] torchrun 退出码 {rc}")
        return 1

    import torch

    problems: list[str] = []
    last = os.path.join(out, "last.pt")
    if not os.path.exists(last):
        problems.append("last.pt 不存在")
    else:
        payload = torch.load(last, map_location="cpu", weights_only=False)
        keys = list(payload["model"].keys())
        if any(k.startswith("module.") for k in keys):
            problems.append("checkpoint 带 module. 前缀（单卡/微调会加载失败）")
        if payload.get("step") != 4:
            problems.append(f"step={payload.get('step')} != 4")
        if "ema" not in payload:
            problems.append("checkpoint 缺少 EMA")
    for name in ("best.pt", "standardizer.json", "ema_model.pt", "final_metrics.json"):
        if not os.path.exists(os.path.join(out, name)):
            problems.append(f"缺少产物 {name}")

    if not args.keep:
        _cleanup([cache, out])
    else:
        print(f"[keep] cache={cache} out={out}")
    if problems:
        for p in problems:
            print("[FAIL]", p)
        return 1
    print(f"[ok  ] DDP 端到端验证通过（{args.nproc} 进程，4 步，检查点/EMA/产物齐全）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
