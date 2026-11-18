#!/usr/bin/env python
# 从 SMILES 生成 PCQ valid 的 3D 构象（RDKit ETKDGv3 + MMFF 优化）。
#
# 背景：PCQM4Mv2 官方 valid 只给 SMILES 和 gap 标签，没有 3D 坐标（官方
# 设定就是推理时只有 SMILES）。DeMol 论文没有披露它的 valid 构象来源，
# 3D 模型社区的通行做法是 RDKit ETKDG 生成 + MMFF 弛豫，本脚本按此实现。
#
# 输入 --smiles-file 支持两种格式：
#   1. 纯文本：每行一个 SMILES（没有 gap 标签，只能用于前向演示）
#   2. CSV：含 smiles 列，可选 gap 或 y 列、可选 idx 列
# 输出与 extract_pcq_subset_sdf.py 同构的 jsonl.gz，可直接喂给
# scripts/preprocess_pcq.py 的 --valid-jsonl。
#
# 用法（A100 上，venv 有 rdkit）：
#   ~/yangsensen/venvs/demol/bin/python scripts/gen_pcq_valid_conformers.py \
#       --smiles-file data/pcq/valid_smiles.csv \
#       --out data/pcq/valid_rdkit.jsonl.gz

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time


def read_smiles_file(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.readline()
        fh.seek(0)
        if "," in head and "smiles" in head.lower():
            for i, row in enumerate(csv.DictReader(fh)):
                smi = row.get("smiles") or row.get("SMILES") or ""
                gap = row.get("gap", row.get("y", ""))
                rows.append({
                    "idx": int(row.get("idx", i) or i),
                    "smiles": smi.strip(),
                    "gap": float(gap) if gap not in ("", None) else None,
                })
        else:
            for i, line in enumerate(fh):
                smi = line.strip()
                if smi:
                    rows.append({"idx": i, "smiles": smi, "gap": None})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles-file", required=True)
    ap.add_argument("--out", required=True, help="输出 jsonl.gz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmff-iters", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
    except ImportError:
        print("[err ] 需要 rdkit（服务器 venv 里有）", file=sys.stderr)
        return 2
    RDLogger.DisableLog("rdApp.*")

    rows = read_smiles_file(args.smiles_file)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[info] {len(rows)} 个 SMILES 待生成构象")

    t0 = time.time()
    ok = failed = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for row in rows:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                failed += 1
                continue
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = args.seed + row["idx"]
            if AllChem.EmbedMolecule(mol, params) != 0:
                failed += 1
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=args.mmff_iters)
            except Exception:
                pass
            conf = mol.GetConformer()
            rec = {
                "position": [list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())],
                "node_class": [a.GetAtomicNum() for a in mol.GetAtoms()],
                "unitcell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "labels": {"object": {"SMILES": row["smiles"], "gap": row["gap"]},
                           "node": None},
                "subset_idx": row["idx"],
                "conformer": "rdkit-etkdgv3+mmff",
            }
            out.write(json.dumps(rec) + "\n")
            ok += 1
            if ok % 5000 == 0:
                rate = ok / max(1e-9, time.time() - t0)
                print(f"       {ok} ok / {failed} failed ({rate:.0f}/s)", flush=True)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[ok  ] {ok} 条构象 -> {args.out} ({size_mb:.1f} MB), 失败 {failed} 条, "
          f"耗时 {(time.time() - t0) / 60:.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
