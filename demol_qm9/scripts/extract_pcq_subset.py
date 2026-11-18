#!/usr/bin/env python
"""按子集名单 csv 从 PCQM4Mv2 原始 jsonl.gz 里抽出带 3D 坐标的记录。

背景：子集名单（如 chemberta2_train_subset_140k.csv）只有 idx/smiles/gap，
没有坐标；而原始 chunk_*.jsonl.gz 里有 position/node_class 但**没有 id 字段**，
所以 idx 按"train chunk 顺序拼接后的行号"（0 基）解释。脚本会抽样比对
csv 的 smiles 与 jsonl 记录里的 labels.object.SMILES，若不一致直接报错退出，
防止名单和数据错位。

用法（在存有原始数据的服务器上跑，纯标准库，无需装任何包）::

    python3 extract_pcq_subset.py \
        --data-dir /mnt/data/yangsensen/data \
        --subset-csv chemberta2_train_subset_140k.csv \
        --out pcq_chemberta2_140k.jsonl.gz

输出还是 jsonl.gz，每条记录在原结构上多一个 ``subset_idx`` 字段（原 idx），
后面预处理脚本可以直接读。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import sys


def load_subset(csv_path: str) -> dict[int, str]:
    """读名单 -> {idx: smiles}。"""
    table: dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        import csv as _csv

        for row in _csv.DictReader(fh):
            table[int(row["idx"])] = row["smiles"]
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="放 chunk_*_train.jsonl.gz 的目录")
    ap.add_argument("--subset-csv", required=True, help="子集名单 csv（含 idx, smiles 列）")
    ap.add_argument("--out", required=True, help="输出 jsonl.gz 路径")
    ap.add_argument("--check", type=int, default=200, help="随机抽样比对 smiles 的条数")
    args = ap.parse_args(argv)

    chunks = sorted(
        f for f in os.listdir(args.data_dir)
        if f.startswith("chunk_") and f.endswith("_train.jsonl.gz")
    )
    if not chunks:
        print(f"[err ] {args.data_dir} 下没有 chunk_*_train.jsonl.gz", file=sys.stderr)
        return 2
    print(f"[info] 发现 {len(chunks)} 个 train chunk: {', '.join(chunks)}")

    wanted = load_subset(args.subset_csv)
    print(f"[info] 名单 {len(wanted)} 条, idx 范围 {min(wanted)}..{max(wanted)}")

    rng = random.Random(0)
    check_idx = set(rng.sample(sorted(wanted), min(args.check, len(wanted))))
    checked = mismatched = 0

    found: dict[int, dict] = {}
    offset = 0
    for name in chunks:
        path = os.path.join(args.data_dir, name)
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                gidx = offset + n
                if gidx in wanted:
                    rec = json.loads(line)
                    smi = rec.get("labels", {}).get("object", {}).get("SMILES")
                    if gidx in check_idx:
                        checked += 1
                        if smi != wanted[gidx]:
                            mismatched += 1
                            print(f"[err ] idx={gidx} SMILES 对不上: "
                                  f"csv={wanted[gidx]!r} jsonl={smi!r}", file=sys.stderr)
                    rec["subset_idx"] = gidx
                    found[gidx] = rec
                n += 1
        print(f"[info] {name}: {n} 行, 累计命中 {len(found)}/{len(wanted)}")
        offset += n
        if mismatched:
            print("[err ] smiles 抽样比对失败，idx 很可能不是按 chunk 顺序拼接的行号，"
                  "请把报错贴回来", file=sys.stderr)
            return 1

    missing = len(wanted) - len(found)
    if missing:
        print(f"[err ] 还有 {missing} 条没找到（chunk 不全或 idx 口径不对）", file=sys.stderr)
        return 1

    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for gidx in sorted(found):
            out.write(json.dumps(found[gidx]) + "\n")
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[ok  ] {len(found)} 条 -> {args.out} ({size_mb:.1f} MB), "
          f"抽样比对 {checked} 条全部一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
