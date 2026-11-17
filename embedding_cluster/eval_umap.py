# -*- coding: utf-8 -*-
# UMAP + sklearn-HDBSCAN 在 5000 模拟数据上的效果评估
# 指标: 簇数 vs 真值12、ARI/NMI、噪声率、picked 是否精确等于 target、真簇覆盖数、有无重复
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_subset as cs

true = pd.read_csv("test_data/test_meta.csv")["true_cluster"].to_numpy()
X_raw = cs.load_embeddings("test_data/test_emb.npy")
TARGET = 500

configs = [
    dict(pca=100, ncomp=15, mcs=30, ms=5),
    dict(pca=100, ncomp=15, mcs=50, ms=5),
    dict(pca=100, ncomp=15, mcs=50, ms=10),
    dict(pca=100, ncomp=30, mcs=50, ms=10),
    dict(pca=0, ncomp=15, mcs=50, ms=10),
]

rows = []
for cfg in configs:
    t0 = time.time()
    X_used, _ = cs.maybe_pca(X_raw, cfg["pca"], 42, "sklearn")
    red = cs.Reducer("sklearn", 42, n_components=cfg["ncomp"]).fit(X_used)
    clus = cs.Clusterer("sklearn", cfg["mcs"], cfg["ms"], 42).fit(red.embedding_)
    lab = clus.labels_
    n_cl = len(np.unique(lab[lab >= 0]))
    picked, meta = cs.select_subset(red.embedding_, lab, clus.strength_, TARGET)
    tp = true[picked]
    rows.append(dict(
        pca=cfg["pca"], ncomp=cfg["ncomp"], mcs=cfg["mcs"], ms=cfg["ms"],
        clusters=n_cl, noise=round(float((lab == -1).mean()), 3),
        ARI=round(float(adjusted_rand_score(true, lab)), 3) if n_cl > 1 else None,
        NMI=round(float(normalized_mutual_info_score(true, lab)), 3) if n_cl > 1 else None,
        picked=len(picked), dup=int(len(picked) - len(set(picked.tolist()))),
        true_covered=len(set(tp[tp >= 0])),
        n_outlier=sum(1 for m in meta if m[1] == "outlier"),
        sec=round(time.time() - t0, 1),
    ))
    print("done", cfg, rows[-1], flush=True)

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
