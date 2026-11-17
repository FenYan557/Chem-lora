# -*- coding: utf-8 -*-
"""
生成 5000 条模拟 embedding (768维, 类 MoLFormer 输出) 用于本地验证 cluster_subset.py
结构: 12 个高斯团 (大小 150~900 不等) + 5% 均匀散布噪声, 最后 L2 归一化
输出: test_data/test_emb.npy, test_data/test_meta.csv
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N_TOTAL = 5000
DIM = 768
N_CLUSTERS = 12
NOISE_FRAC = 0.05
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

rng = np.random.default_rng(SEED)
os.makedirs(OUT, exist_ok=True)

# 1) 簇大小: 一大二中的长尾分布, 模拟真实分子库
sizes = np.array([900, 700, 600, 500, 450, 400, 350, 300, 250, 200, 150, 150])
n_noise = int(N_TOTAL * NOISE_FRAC)
sizes = (sizes * (N_TOTAL - n_noise) / sizes.sum()).astype(int)
sizes[-1] += (N_TOTAL - n_noise) - sizes.sum()

rows, true_labels = [], []
for c, n in enumerate(sizes):
    center = rng.normal(size=DIM).astype(np.float32)
    center /= np.linalg.norm(center)
    spread = 0.12 + 0.01 * c                      # 各簇紧致度略有差异
    pts = center[None, :] + spread * rng.normal(size=(n, DIM)).astype(np.float32)
    rows.append(pts)
    true_labels.append(np.full(n, c, dtype=np.int32))

# 2) 噪声: 均匀散布在整个球面上
noise = rng.normal(size=(n_noise, DIM)).astype(np.float32)
noise /= np.linalg.norm(noise, axis=1, keepdims=True) * 3
rows.append(noise)
true_labels.append(np.full(n_noise, -1, dtype=np.int32))

X = np.concatenate(rows, axis=0)
true_labels = np.concatenate(true_labels)
X /= np.linalg.norm(X, axis=1, keepdims=True)     # 与真实 embedding 一样 L2 归一化

# 3) 打乱顺序, 并造 meta (行号对齐)
perm = rng.permutation(N_TOTAL)
X, true_labels = X[perm], true_labels[perm]
meta = pd.DataFrame({
    "smiles": [f"C{i}test_smiles_{i:05d}" for i in range(N_TOTAL)],
    "true_cluster": true_labels,
})

np.save(os.path.join(OUT, "test_emb.npy"), X.astype(np.float32))
meta.to_csv(os.path.join(OUT, "test_meta.csv"), index=False)
print(f"saved {X.shape} emb + meta -> {OUT}")
print("cluster sizes:", np.bincount(true_labels[true_labels >= 0]).tolist(),
      "noise:", int((true_labels == -1).sum()))
