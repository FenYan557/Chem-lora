# chem-lora

分子机器学习研究仓库，目前包含两个子项目：

| 子项目 | 目录 | 内容 |
| --- | --- | --- |
| PCQM4Mv2 子集蒸馏 | `embedding_cluster/` | 三种分子 embedding（MoLFormer-XL / ChemBERTa-2 / Morgan 指纹）→ UMAP 降维 → HDBSCAN 密度聚类 → 每簇选代表点，从 337 万分子中蒸馏代表性子集 |
| DeMol 复现 | `demol_qm9/` | DeMol（Liu et al., 2026）原子+键双通道分子 Transformer 的 PyTorch 复现，QM9 预训练 / 微调 |

## 目录结构

```
chem-lora/
├── docs/
│   └── dataset_benchmarks.csv   # QM9 / PCQM4Mv2 / Molecule3D / Alchemy 等数据集基准调研
├── embedding_cluster/           # 子项目 1：embedding 聚类子集蒸馏
│   ├── README.md                # 子项目说明
│   ├── docs/                    # 交接文档、采样合理性论证、实现细节
│   ├── server_scripts/          # 服务器端全流程脚本（提嵌入 / 聚类 / 重采样 / QA）
│   └── *.py                     # 本地脚本（cluster_subset.py 是主脚本）
└── demol_qm9/                   # 子项目 2：DeMol 复现（有自己的 README）
    ├── demol/                   # 库代码
    └── scripts/                 # 数据、训练、验证、ModelScope 搬运入口
```

## 数据说明

数据与产物**不进仓库**（见 `.gitignore`）：

- `embedding_cluster` 的 embedding、聚类结果、子集 CSV 通过 `server_scripts/` 在服务器上重新生成，流程见 `embedding_cluster/docs/交接文档.md`。
- `demol_qm9` 的 QM9 数据获取见 `demol_qm9/README.md`（figshare 直接下载，或本地特征化后经 ModelScope 中转）。
