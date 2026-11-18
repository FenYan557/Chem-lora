#!/usr/bin/env python
"""从官方 PCQM4Mv2 训练集 SDF（pcqm4m-v2-train.sdf）按 SMILES 抽取子集。

为什么按 SMILES 而不是 idx：子集名单里的 idx 是旧服务器 jsonl 的行号
（train+valid 合并口径，最大 3,452,073），与官方 train SDF（3,378,606 条）
顺序不同，按 idx 会错位。gap 标签直接取名单 csv 的 y 列，不需要官方标签文件。

输出 jsonl.gz 与旧服务器 jsonl 同构（position / node_class / labels.object.*），
多一个 subset_idx 字段；unitcell 官方 SDF 没有，写单位阵占位。

用法（A100 上，venv 里已有 rdkit）::

    ~/yangsensen/venvs/demol/bin/python scripts/extract_pcq_subset_sdf.py \
        --sdf data/pcq/pcqm4m-v2-train.sdf \
        --subset-csv data/pcq/chemberta2_train_subset_140k.csv \
        --out data/pcq/pcq_chemberta2_140k.jsonl.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import defaultdict, deque

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    # isomericSmiles=False：剥离立体化学，避免 csv 的 @ 与 SDF 无手性标记不一致
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf", required=True, help="官方 pcqm4m-v2-train.sdf 路径")
    ap.add_argument("--subset-csv", required=True, help="名单 csv（idx, smiles, y 列）")
    ap.add_argument("--out", required=True, help="输出 jsonl.gz")
    args = ap.parse_args(argv)

    wanted: dict[str, deque] = defaultdict(deque)  # canonical -> deque[(idx, gap)]
    bad = 0
    with open(args.subset_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            csmi = canonical(row["smiles"])
            if csmi is None:
                bad += 1
                continue
            wanted[csmi].append((int(row["idx"]), float(row["y"])))
    n_target = sum(len(v) for v in wanted.values())
    print(f"[info] 名单 {n_target} 条（canonical SMILES {len(wanted)} 种，"
          f"无法解析 {bad} 条）")

    t0 = time.time()
    found = 0
    suppl = Chem.SDMolSupplier(args.sdf, removeHs=False)
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for i, mol in enumerate(suppl):
            if (i + 1) % 200000 == 0:
                print(f"[info] 已扫 {i + 1} 条, 命中 {found}/{n_target}, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if mol is None or mol.GetNumConformers() == 0:
                continue
            # SDMolSupplier(removeHs=False) 读出的 mol 带显式 H，
            # 直接 MolToSmiles 会得到 [H] 标记的串，与名单（隐式 H）永远对不上；
            # 匹配键去掉 H，坐标/原子仍从原始 mol（含 H）导出。
            csmi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False),
                                    isomericSmiles=False)
            queue = wanted.get(csmi)
            if not queue:
                continue
            idx, gap = queue.popleft()
            conf = mol.GetConformer()
            rec = {
                "position": [list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())],
                "node_class": [a.GetAtomicNum() for a in mol.GetAtoms()],
                "unitcell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "labels": {"object": {"SMILES": csmi, "gap": gap}, "node": None},
                "subset_idx": idx,
            }
            out.write(json.dumps(rec) + "\n")
            found += 1
            if found >= n_target:
                break

    missing = n_target - found
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[ok  ] 命中 {found}/{n_target} -> {args.out} ({size_mb:.1f} MB), "
          f"耗时 {(time.time() - t0) / 60:.1f} min")
    if missing:
        print(f"[warn] {missing} 条名单分子在 SDF 里没匹配上"
              "（约 4% 的 SDF 键感知与 SMILES 图不一致，属预期损耗）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
