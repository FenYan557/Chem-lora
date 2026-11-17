# embedding_cluster —— PCQM4Mv2 子集蒸馏

从 PCQM4Mv2 的 3,452,095 个分子中挑出代表性子集（目标 5 万），路线：
**embedding → (可选 PCA) → UMAP 降维 → HDBSCAN 密度聚类 → 每簇挑代表点**。

三种互为对比的 embedding：MoLFormer-XL（768d）/ ChemBERTa-2-77M-MTR（384d）/ Morgan 指纹（2048 bit）。

## 文档（先看 docs/交接文档.md）

- `docs/交接文档.md` —— 现状速览、整体流水线、服务器操作手册
- `docs/采样合理性与交接报告.md` —— 采样方法的合理性论证与实验依据
- `docs/HDBSCAN_子集筛选_实现逻辑.md` —— 聚类选点的实现细节

## 本地脚本（本目录）

- `cluster_subset.py` —— 主脚本：embedding → 降维 → 聚类 → 选代表点（cuml GPU / sklearn CPU 双后端自适应）
- `analyze_subset.py` / `eval_umap.py` / `eval_test_configs.py` —— 子集分析与评估
- `make_test_data.py` / `time_one_config.py` / `check_env.py` / `probe_*.py` —— 测试数据、计时、环境与小探针

## 服务器脚本（server_scripts/）

提嵌入（`embed_*.py`）、聚类（`cluster_*.py` / `cluster_full.sh`）、子集重采样（`resample_train_subset.py`）、
QA（`qa_pilot_subsets.py` / `scaffold_qa.py` / `validate_emb.py`）、分片合并（`merge_shards.py`）等。

> 注意：shell 脚本内的路径是**服务器布局**（`/mnt/data/yangsensen/embedding/...`），
> 与本仓库目录结构无关；部署到服务器时按 `docs/交接文档.md` 的说明安放。
