# -*- coding: utf-8 -*-
# 检查本地 5000 样本试跑产出的子集质量
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

meta = pd.read_csv("test_data/test_meta.csv")
sub = pd.read_csv("test_subsets/test_emb_subset.csv")
labels_all = np.load("test_subsets/test_emb_labels_all.npy")
cfg = json.load(open("test_subsets/test_emb_run_config.json", encoding="utf-8"))

true = meta["true_cluster"].to_numpy()
sub_true = sub["true_cluster"].to_numpy()

print("run_config:", {k: cfg[k] for k in ("backend", "reducer", "N", "dim",
                                          "n_clusters", "noise_rate", "picked", "n_outlier")})
print()
print("子集大小:", len(sub), " 目标:", cfg["target"], " 一致:", len(sub) == cfg["target"])
print("子集内重复行:", len(sub) - len(sub.drop_duplicates(subset=["smiles"])))
print("覆盖真簇数:", len(set(sub_true[sub_true >= 0])), "/ 12   含噪声点:",
      int((sub_true == -1).sum()))
print()
print("HDBSCAN 簇 vs 真簇:  ARI=%.3f  NMI=%.3f" % (
    adjusted_rand_score(true, labels_all),
    normalized_mutual_info_score(true, labels_all)))
print()
per = sub.groupby("cluster").size().sort_values(ascending=False)
print("各簇选中数 (sqrt 配额, 大簇多给):")
print(per.to_string())
print()
print("每个真簇在全库 vs 子集中的占比:")
t = pd.crosstab(meta["true_cluster"], "all")
t["subset"] = pd.crosstab(sub["true_cluster"], "subset").reindex(t.index).fillna(0).astype(int)
t["subset_rate"] = (t["subset"] / t["all"]).round(3)
print(t.to_string())
print()
print("role 分布:", sub["role"].value_counts().to_dict())
print("strength 分布: min=%.3f median=%.3f max=%.3f" % (
    sub["strength"].min(), sub["strength"].median(), sub["strength"].max()))
