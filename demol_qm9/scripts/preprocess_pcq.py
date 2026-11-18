#!/usr/bin/env python
# 把 PCQ 的 jsonl(.gz) 特征化成 DeMol 缓存。
#
# 输入 jsonl 可以来自两条路（结构同构）：
#   scripts/extract_pcq_subset_sdf.py  官方 SDF 抽出的 train 子集（真 DFT 构象）
#   scripts/gen_pcq_valid_conformers.py  valid SMILES 经 RDKit 生成的构象
#
# 用法（A100 上）：
#   ~/yangsensen/venvs/demol/bin/python scripts/preprocess_pcq.py \
#       --train-jsonl data/pcq/train_subset_140k.jsonl.gz \
#       --valid-jsonl data/pcq/valid_rdkit.jsonl.gz \
#       --cache-dir data/pcq_cache
#
# train 记录写在前、valid 在后，split.npy 里 train=0 / val=1（无 test）。
# 132k 分子串行约 5-10 分钟（rdkit 键感知）。peak memory 约 2 GB。

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demol.features import featurize_molecule, rdkit_available  # noqa: E402
from demol.pcq import iter_pcq_jsonl, write_pcq_split  # noqa: E402
from demol.qm9 import write_cache  # noqa: E402


def featurize_stream(paths, prefer_rdkit: bool, limit: int | None, tag: str) -> list[dict]:
    records: list[dict] = []
    skipped = 0
    t0 = time.time()
    stream = iter_pcq_jsonl(paths)
    if limit is not None:
        stream = itertools.islice(stream, limit)
    for i, rec in enumerate(stream):
        feats = featurize_molecule(rec["pos"], rec["z"], prefer_rdkit=prefer_rdkit)
        if feats.bond_pairs.shape[0] == 0:
            skipped += 1
            continue
        records.append({
            "index": rec["index"],
            "feats": feats,
            "targets": np.array([rec["gap"]], dtype=np.float32),
        })
        if (i + 1) % 5000 == 0:
            rate = len(records) / max(1e-9, time.time() - t0)
            print(f"       [{tag}] {i + 1:7d} scanned, {len(records)} featurised "
                  f"({rate:5.0f}/s, skipped {skipped})", flush=True)
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True,
                    help="训练子集 jsonl(.gz)，可用逗号传多个文件")
    ap.add_argument("--valid-jsonl", default=None,
                    help="验证集 jsonl(.gz)；不传则全部记为 train")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--backend", choices=("auto", "rdkit", "geometry"), default="auto")
    ap.add_argument("--limit", type=int, default=None, help="每个输入只处理前 N 条（调试用）")
    args = ap.parse_args(argv)

    use_rdkit = args.backend == "rdkit" or (args.backend == "auto" and rdkit_available())
    print(f"[info] featurisation backend: {'rdkit' if use_rdkit else 'geometry'}")

    t0 = time.time()
    train_paths = [p.strip() for p in args.train_jsonl.split(",") if p.strip()]
    train_records = featurize_stream(train_paths, use_rdkit, args.limit, "train")
    n_train = len(train_records)
    print(f"[info] train: {n_train} molecules ({time.time() - t0:.0f} s)")

    val_records: list[dict] = []
    if args.valid_jsonl:
        val_paths = [p.strip() for p in args.valid_jsonl.split(",") if p.strip()]
        val_records = featurize_stream(val_paths, use_rdkit, args.limit, "valid")
    n_val = len(val_records)
    print(f"[info] valid: {n_val} molecules")

    os.makedirs(args.cache_dir, exist_ok=True)
    stats = write_cache(train_records + val_records, args.cache_dir)
    write_pcq_split(args.cache_dir, n_train, n_val)

    gaps = np.array([r["targets"][0] for r in train_records], dtype=np.float64)
    stats.update({
        "backend": "rdkit" if use_rdkit else "geometry",
        "n_targets": 1,
        "targets": [{"name": "gap", "unit": "eV"}],
        "gap_ev": {"mean": round(float(gaps.mean()), 4), "std": round(float(gaps.std()), 4),
                   "min": round(float(gaps.min()), 4), "max": round(float(gaps.max()), 4)},
        "n_train": n_train,
        "n_val": n_val,
    })
    with open(os.path.join(args.cache_dir, "cache_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[ok  ] cache -> {args.cache_dir}: train={n_train} val={n_val}, "
          f"gap mean={gaps.mean():.3f} eV, 耗时 {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
