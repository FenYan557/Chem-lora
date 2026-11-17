# -*- coding: utf-8 -*-
"""单配置计时: 定位 UMAP 后端下哪一步慢"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cluster_subset as cs

t0 = time.time()
X = cs.load_embeddings("test_data/test_emb.npy")
print(f"load {X.shape} {time.time()-t0:.1f}s", flush=True)

t = time.time()
X_used, _ = cs.maybe_pca(X, 100, 42, "sklearn")
print(f"pca -> {X_used.shape} {time.time()-t:.1f}s", flush=True)

t = time.time()
red = cs.Reducer("sklearn", 42, n_components=15).fit(X_used)
print(f"reduce {red.kind} -> {red.embedding_.shape} {time.time()-t:.1f}s", flush=True)

t = time.time()
clus = cs.Clusterer("sklearn", 50, 10, 42).fit(red.embedding_)
lab = clus.labels_
print(f"hdbscan clusters={len(np.unique(lab[lab>=0]))} noise={(lab==-1).mean():.2%} {time.time()-t:.1f}s", flush=True)

t = time.time()
picked, meta = cs.select_subset(red.embedding_, lab, clus.strength_, 500)
print(f"select picked={len(picked)} {time.time()-t:.1f}s", flush=True)
