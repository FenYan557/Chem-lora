# -*- coding: utf-8 -*-
"""在 5000 条模拟数据上对比不同配置的聚类质量 (ARI/NMI/簇数/噪声率/选中数)"""
import itertools
import subprocess
import sys
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_subset as cs

META = "test_data/test_meta.csv"
EMB = "test_data/test_emb.npy"
true = pd.read_csv(META)["true_cluster"].to_numpy()
X_raw = cs.load_embeddings(EMB)
N = len(X_raw)

configs = []
for pca_dim, n_comp, mcs, ms in itertools.product(
        [0, 50, 100], [15, 30], [30, 50, 100], [5, 10]):
    configs.append(dict(pca_dim=pca_dim, n_comp=n_comp, mcs=mcs, ms=ms))

rows = []
for cfg in configs:
    X_used, _ = cs.maybe_pca(X_raw, cfg["pca_dim"], 42, "sklearn")
    red = cs.Reducer("sklearn", 42, n_components=cfg["n_comp"]).fit(X_used)
    clus = cs.Clusterer("sklearn", cfg["mcs"], cfg["ms"], 42).fit(red.embedding_)
    lab = clus.labels_
    n_cl = len(np.unique(lab[lab >= 0]))
    noise = float((lab == -1).mean())
    ari = adjusted_rand_score(true, lab) if n_cl > 1 else float("nan")
    nmi = normalized_mutual_info_score(true, lab) if n_cl > 1 else float("nan")
    # 选中数量是否严格 == target
    picked, _ = cs.select_subset(red.embedding_, lab, clus.strength_, 500)
    rows.append(dict(pca=cfg["pca_dim"], ncomp=cfg["n_comp"], mcs=cfg["mcs"],
                     ms=cfg["ms"], clusters=n_cl, noise=round(noise, 3),
                     ARI=round(float(ari), 3) if ari == ari else None,
                     NMI=round(float(nmi), 3) if nmi == nmi else None,
                     picked=len(picked),
                     true_cluster_covered=len(set(true[picked][true[picked] >= 0]))))

df = pd.DataFrame(rows).sort_values("ARI", ascending=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
