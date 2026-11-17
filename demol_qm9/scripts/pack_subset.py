#!/usr/bin/env python
"""把 QM9 缓存切成子集，打包成可以上传 ModelScope 的目录。

训练要在租来的大显卡上跑，而服务器上重新下载 1.6 GB 原始 QM9、再跑一遍
featurisation 既慢又容易失败（figshare 在国内经常连不上）。所以本地一次做好：

    python scripts/preprocess_qm9.py --raw-dir data/qm9/dsgdb9nsd.xyz \
                                     --cache-dir data/qm9_cache
    python scripts/pack_subset.py --cache-dir data/qm9_cache \
                                  --out-dir data/qm9_subset_20k --size 20000
    python scripts/upload_modelscope.py --local-dir data/qm9_subset_20k \
                                        --repo-id <你的用户名>/demol-qm9-20k

产出的目录本身就是一个合法的 DeMol 缓存（``.npy`` + ``split.npy`` +
``cache_stats.json``），可以直接被 ``QM9Dataset`` 打开；另外多一个
``manifest.json``，记录每个文件的 sha256、每个数组的 shape/dtype、以及打包时的
环境信息。服务器端 ``scripts/download_modelscope.py`` 会按它逐文件校验。

``--select`` 的三种取法：

* ``stratified``（默认）按源缓存 train/val/test 的比例分层抽样，子集里三块都还在，
  服务器上可以直接跑完整的"训练 + 验证 + 测试"流程；
* ``random`` 纯随机抽；
* ``head`` 取前 N 个，最快，只适合验证通路。

不给 ``--size`` / ``--frac`` 就是整份缓存原样打包（QM9 全量的缓存只有几十 MB，
通常直接全传更省事）。
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

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

from demol import hubio  # noqa: E402
from demol.qm9 import (  # noqa: E402
    N_TEST,
    N_VAL,
    QM9Dataset,
    load_cache_arrays,
    make_split,
    slice_cache,
)

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------
# 选行
# --------------------------------------------------------------------------
def load_source_split(cache_dir: str, n_total: int) -> tuple[np.ndarray, bool]:
    """读源缓存的 split.npy；没有就全当训练集（返回 False 表示是兜底）。"""
    path = os.path.join(cache_dir, "split.npy")
    if not os.path.exists(path):
        return np.zeros(n_total, dtype=np.int8), False
    split = np.load(path)
    if split.shape[0] != n_total:
        raise ValueError(f"split.npy 长度 {split.shape[0]} != 分子数 {n_total}")
    return split.astype(np.int8), True


def allocate(counts: np.ndarray, size: int) -> np.ndarray:
    """按比例把 size 个名额分到各层，最大余数法，且不超过各层容量。"""
    total = int(counts.sum())
    if size >= total:
        return counts.astype(np.int64).copy()
    exact = counts.astype(np.float64) / total * size
    quota = np.floor(exact).astype(np.int64)
    quota = np.minimum(quota, counts.astype(np.int64))
    remaining = size - int(quota.sum())
    order = np.argsort(-(exact - quota))
    guard = 0
    while remaining > 0 and guard < 100 * len(counts) + 100:
        j = int(order[guard % len(order)])
        if quota[j] < counts[j]:
            quota[j] += 1
            remaining -= 1
        guard += 1
    return quota


def select_rows(split: np.ndarray, size: int, mode: str, seed: int) -> np.ndarray:
    """返回排序去重后的行号。"""
    n = int(split.shape[0])
    rng = np.random.default_rng(seed)
    if mode == "head":
        return np.arange(min(size, n), dtype=np.int64)
    if mode == "random":
        if size >= n:
            return np.arange(n, dtype=np.int64)
        return np.sort(rng.choice(n, size=size, replace=False)).astype(np.int64)
    if mode != "stratified":
        raise ValueError(f"未知的 --select：{mode}")
    counts = np.bincount(split.astype(np.int64), minlength=len(SPLIT_NAMES))
    quota = allocate(counts[: len(SPLIT_NAMES)], min(size, n))
    picked: list[np.ndarray] = []
    for sid in range(len(SPLIT_NAMES)):
        pool = np.flatnonzero(split == sid)
        take = int(min(quota[sid], pool.size))
        if take:
            picked.append(np.sort(rng.choice(pool, size=take, replace=False)))
    rows = np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)
    rows = np.unique(rows).astype(np.int64)
    if rows.size < size:  # 分层配额被各层容量卡住时，从剩下的里补齐
        pool = np.setdiff1d(np.arange(n, dtype=np.int64), rows)
        need = min(size - int(rows.size), int(pool.size))
        if need > 0:
            rows = np.unique(np.concatenate([rows, rng.choice(pool, size=need, replace=False)]))
    return rows.astype(np.int64)


def build_subset_split(mode: str, src_split: np.ndarray, rows: np.ndarray, seed: int) -> np.ndarray:
    """子集的 split.npy。"""
    n = int(rows.size)
    if mode == "keep":
        return src_split[rows].astype(np.int8)
    if mode == "none":
        return np.zeros(n, dtype=np.int8)
    if mode != "proportional":
        raise ValueError(f"未知的 --split：{mode}")
    # 论文比例：10000 val / 10831 test / 其余训练，总共 133885 个分子
    n_val = int(round(n * N_VAL / 133885))
    n_test = int(round(n * N_TEST / 133885))
    n_val = max(1, min(n_val, n - 2))
    n_test = max(1, min(n_test, n - 1 - n_val))
    return make_split(n, seed=seed, n_val=n_val, n_test=n_test).astype(np.int8)


# --------------------------------------------------------------------------
# 自检：子集里的分子必须和源缓存逐字段相同
# --------------------------------------------------------------------------
def self_check(cache_dir: str, out_dir: str, rows: np.ndarray, n_probe: int) -> tuple[list[str], int]:
    """抽查若干分子，逐字段比对源缓存与子集；返回 (问题列表, 实际抽查数)。"""
    problems: list[str] = []
    src = QM9Dataset(cache_dir, split_ids=(0, 1, 2))
    sub = QM9Dataset(out_dir, split_ids=(0, 1, 2))
    try:
        if len(sub) != len(rows):
            problems.append(f"子集分子数 {len(sub)} != 期望 {len(rows)}")
        if not np.array_equal(np.asarray(sub.arrays["index"]), np.asarray(src.arrays["index"])[rows]):
            problems.append("子集 index 数组与源缓存对不上")
        n_probe = max(1, min(n_probe, len(sub)))
        step = max(1, len(sub) // n_probe)
        probes = list(range(0, len(sub), step))[:n_probe]
        for k in probes:
            a = src[int(rows[k])]
            b = sub[k]
            if set(a) != set(b):
                problems.append(f"分子 {k} 字段不一致：{sorted(set(a) ^ set(b))}")
                break
            for key in a:
                if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                    problems.append(f"分子 {k} 字段 {key} 与源缓存不同")
                    break
            if problems:
                break
        return problems, len(probes)
    finally:
        sub.close()
        src.close()


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache-dir", default=os.path.join("data", "qm9_cache"),
                    help="源缓存目录（preprocess_qm9.py 的输出）")
    ap.add_argument("--out-dir", required=True, help="子集输出目录（也是上传目录）")
    size_group = ap.add_mutually_exclusive_group()
    size_group.add_argument("--size", type=int, default=None, help="子集分子数（默认=整份缓存）")
    size_group.add_argument("--frac", type=float, default=None, help="子集比例，例如 0.15")
    ap.add_argument("--select", choices=("stratified", "random", "head"), default="stratified",
                    help="怎么选（默认 stratified：按 train/val/test 比例分层抽样）")
    ap.add_argument("--split", choices=("keep", "proportional", "none"), default="keep",
                    help="子集的 split.npy 怎么写（默认 keep：沿用源缓存的标签）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archive", choices=("none", "tar.gz", "zip"), default="none",
                    help="额外压成单文件（默认 none：直接传目录）")
    ap.add_argument("--check-mols", type=int, default=64,
                    help="自检比对多少个分子（默认 64，0 = 跳过自检）")
    ap.add_argument("--force", action="store_true", help="out-dir 非空时先清空")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.cache_dir):
        print(f"[err ] 源缓存不存在：{args.cache_dir}", file=sys.stderr)
        print("       先跑 scripts/preprocess_qm9.py 生成缓存", file=sys.stderr)
        return 2

    started = time.time()
    arrays = load_cache_arrays(args.cache_dir)
    n_total = int(len(arrays["index"]))
    split, had_split = load_source_split(args.cache_dir, n_total)
    for arr in arrays.values():
        mmap = getattr(arr, "_mmap", None)
        if mmap is not None:
            mmap.close()
    if not had_split:
        print("[warn] 源缓存没有 split.npy，按「全部训练集」处理（--split keep 会全是 0）")

    if args.frac is not None:
        if not 0 < args.frac <= 1:
            print(f"[err ] --frac 要在 (0, 1] 之间，收到 {args.frac}", file=sys.stderr)
            return 2
        size = max(1, int(round(args.frac * n_total)))
    else:
        size = n_total if args.size is None else max(1, min(args.size, n_total))

    rows = select_rows(split, size, args.select, args.seed)
    subset_split = build_subset_split(args.split, split, rows, args.seed)
    counts = np.bincount(subset_split.astype(np.int64), minlength=3)
    print(f"[info] 源缓存 {n_total} 个分子 -> 子集 {len(rows)} 个"
          f"（--select {args.select}，seed {args.seed}）")
    print(f"[info] 子集 split（{args.split}）：train={counts[0]} val={counts[1]} test={counts[2]}")

    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        if not args.force:
            print(f"[err ] {args.out_dir} 非空；加 --force 覆盖", file=sys.stderr)
            return 2
        hubio_clean(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    stats = slice_cache(args.cache_dir, args.out_dir, rows, split=subset_split)
    print(f"[ok  ] 已切出 {stats['n_molecules']} 个分子 / {stats['n_atoms']} 个原子 / "
          f"{stats['n_bonds']} 条键（max_atoms={stats['max_atoms']}, max_bonds={stats['max_bonds']}）")

    n_probe = 0
    if args.check_mols > 0:
        problems, n_probe = self_check(args.cache_dir, args.out_dir, rows, args.check_mols)
        if problems:
            for p in problems:
                print(f"[err ] 自检失败：{p}", file=sys.stderr)
            return 1
        print(f"[ok  ] 自检：抽查 {n_probe} 个分子，所有字段与源缓存逐元素相同")

    manifest = hubio.build_manifest(args.out_dir, extra={
        "subset": {
            "source_cache": os.path.abspath(args.cache_dir),
            "source_molecules": n_total,
            "select": args.select,
            "requested_size": size,
            "actual_size": int(len(rows)),
            "seed": args.seed,
            "split_mode": args.split,
            "split_counts": {
                "train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])
            },
            "source_had_split": bool(had_split),
            "self_checked_molecules": int(n_probe),
            "rows_head": [int(r) for r in rows[:8]],
            "qm9_index_head": [int(v) for v in np.asarray(
                np.load(os.path.join(args.out_dir, "index.npy"), mmap_mode="r"))[:8]],
        },
    })
    hubio.write_manifest(args.out_dir, manifest)
    print(f"[ok  ] manifest：{manifest['n_files']} 个文件，"
          f"{hubio.human_size(manifest['total_bytes'])}，sha256 已登记")

    if args.archive != "none":
        ext = "zip" if args.archive == "zip" else "tar.gz"
        archive_path = os.path.join(
            os.path.dirname(os.path.abspath(args.out_dir)) or ".",
            f"{os.path.basename(os.path.abspath(args.out_dir))}.{ext}",
        )
        info = hubio.make_archive(args.out_dir, archive_path, args.archive)
        manifest["archive"] = info
        hubio.write_manifest(args.out_dir, manifest)
        print(f"[ok  ] 归档：{archive_path}（{hubio.human_size(info['bytes'])}，"
              f"{info['seconds']}s，sha256 {info['sha256'][:12]}…）")

    print(f"[ok  ] 用时 {time.time() - started:.1f}s，子集在 {args.out_dir}")
    print()
    print("下一步（上传到 ModelScope）：")
    print(f"  python scripts/upload_modelscope.py --local-dir {args.out_dir} "
          "--repo-id <你的用户名>/demol-qm9-subset")
    print("服务器端（下载 + 校验）：")
    print("  python scripts/download_modelscope.py --repo-id <你的用户名>/demol-qm9-subset "
          f"--cache-dir {args.out_dir}")
    return 0


def hubio_clean(path: str) -> None:
    """清空目录内容但保留目录本身（Windows 上被 mmap 锁住的文件会报错）。"""
    import shutil

    for name in os.listdir(path):
        target = os.path.join(path, name)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                os.remove(target)
            except OSError as exc:
                print(f"[warn] 删不掉 {target}：{exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
