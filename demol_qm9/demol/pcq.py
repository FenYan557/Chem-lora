"""PCQM4Mv2 数据接入：jsonl 解析、缓存构建、词表降级。

数据链路：

    train_subset_140k.jsonl.gz（extract_pcq_subset_sdf.py 从官方 SDF 抽取；
    gen_pcq_valid_conformers.py 从 SMILES 生成构象，同构）
    -> iter_pcq_jsonl  ->  featurize_molecule  ->  write_cache (demol.qm9)
    得到与 QM9 同构的 .npy 缓存，唯一区别是 targets 只有 gap 一列（eV），
    split.npy 里 train=0 / val=1，没有 test。

零样本评估（QM9 预训练检查点直接测 PCQ）时词表对不上：检查点的
atom_field_sizes 是旧 6 类词表，而 PCQ 缓存按扩展后的 14 类词表编码。
VocabDowngradeDataset 把新元素类别（>5）降级回 OTHER=5，让旧检查点能
跑通（代价是 S/Cl 等被当成"未知元素"）。
"""

from __future__ import annotations

import gzip
import json
import os
import time
from typing import Iterator

import numpy as np
import torch

from .features import OTHER_CLASS, featurize_molecule
from .qm9 import QM9Dataset

__all__ = [
    "iter_pcq_jsonl",
    "featurize_pcq_records",
    "write_pcq_split",
    "VocabDowngradeDataset",
    "PCQ_GAP_UNIT",
]

# PCQM4Mv2 官方标签单位是 eV；QM9 管线的 gap 是 meV，跨数据集换算时 /1000。
PCQ_GAP_UNIT = "eV"


def iter_pcq_jsonl(paths: "str | list[str]") -> Iterator[dict]:
    """逐条读取 jsonl(.gz)。记录结构（与旧服务器 jsonl 同构）：
    position (N,3)、node_class (N,) 原子序数、labels.object.SMILES、
    labels.object.gap、可选 subset_idx。
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                obj = rec.get("labels", {}).get("object", {})
                gap = obj.get("gap")
                pos = np.asarray(rec["position"], dtype=np.float32)
                z = np.asarray(rec["node_class"], dtype=np.int64)
                if pos.shape[0] != z.shape[0] or pos.shape[0] == 0:
                    continue
                yield {
                    "index": int(rec.get("subset_idx", i)),
                    "smiles": obj.get("SMILES", ""),
                    "gap": float(gap) if gap is not None else float("nan"),
                    "pos": pos,
                    "z": z,
                }


def featurize_pcq_records(
    paths: "str | list[str]",
    prefer_rdkit: bool = True,
    progress_every: int = 5000,
) -> list[dict]:
    """jsonl -> write_cache 记录列表（targets 宽度 1，单位 eV）。

    特征化与 QM9 管线完全一致（rdkit 键感知优先，几何后端兜底），保证
    预训练/微调的特征同分布。无键分子（m=0）跳过，DeMol 的键通道不能为空。
    """
    records: list[dict] = []
    skipped = 0
    t0 = time.time()
    for i, rec in enumerate(iter_pcq_jsonl(paths)):
        feats = featurize_molecule(rec["pos"], rec["z"], prefer_rdkit=prefer_rdkit)
        if feats.bond_pairs.shape[0] == 0:
            skipped += 1
            continue
        records.append({
            "index": rec["index"],
            "feats": feats,
            "targets": np.array([rec["gap"]], dtype=np.float32),
        })
        if (i + 1) % progress_every == 0:
            rate = len(records) / max(1e-9, time.time() - t0)
            print(f"       {i + 1:7d} scanned, {len(records)} featurised "
                  f"({rate:5.0f}/s, skipped {skipped})", flush=True)
    return records


def write_pcq_split(cache_dir: str, n_train: int, n_val: int) -> np.ndarray:
    """train 记录在前、valid 记录在后写入缓存时，split 就是前段 0 后段 1。"""
    split = np.concatenate([
        np.zeros(n_train, dtype=np.int8),
        np.ones(n_val, dtype=np.int8),
    ])
    np.save(os.path.join(cache_dir, "split.npy"), split)
    return split


class VocabDowngradeDataset(torch.utils.data.Dataset):
    """把扩展词表编码的缓存降级到旧 6 类词表（QM9 检查点零样本评估用）。

    降级规则（与 features.py 的 append-only 布局对齐）：atom_class >5 -> 5；
    atom_fields[..., 0] 与 bond_fields[..., 3:5]（类别+1 的字段值）>6 -> 6。
    """

    def __init__(self, base: QM9Dataset):
        self.base = base
        self._atom_cap = OTHER_CLASS            # 5
        self._field_cap = OTHER_CLASS + 1       # 6

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, k: int) -> dict:
        item = dict(self.base[k])
        item["atom_class"] = np.minimum(item["atom_class"], self._atom_cap)
        af = item["atom_fields"].copy()
        af[:, 0] = np.minimum(af[:, 0], self._field_cap)
        item["atom_fields"] = af
        if item["m"]:
            bf = item["bond_fields"].copy()
            bf[:, 3] = np.minimum(bf[:, 3], self._field_cap)
            bf[:, 4] = np.minimum(bf[:, 4], self._field_cap)
            item["bond_fields"] = bf
        return item
