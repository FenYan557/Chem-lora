# -*- coding: utf-8 -*-
"""定位 umap-learn 在 Windows 上是否卡在初始化"""
import time
import numpy as np
import umap

print("umap", umap.__version__, flush=True)
rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 100)).astype(np.float32)

t = time.time()
emb = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=15,
                metric="euclidean", random_state=42).fit_transform(X)
print(f"2000x100 -> {emb.shape}: {time.time()-t:.1f}s", flush=True)
