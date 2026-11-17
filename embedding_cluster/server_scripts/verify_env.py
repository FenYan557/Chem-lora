"""Sanity check for the yangsensen env: torch/CUDA, rdkit, transformers, umap, hdbscan, cuml."""
import sys
import time

print("python:", sys.version.split()[0], sys.executable)

import numpy as np
import scipy
import sklearn
import pandas as pd

print("numpy:", np.__version__, "| scipy:", scipy.__version__,
      "| sklearn:", sklearn.__version__, "| pandas:", pd.__version__)

import torch

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name}, {p.total_memory/1024**3:.1f} GiB, sm_{p.major}{p.minor}")
    x = torch.randn(1024, 1024, device="cuda:0")
    y = (x @ x).sum().item()
    print("  matmul on cuda:0 ok, sum=", round(y, 2))

import transformers
from rdkit import Chem

print("transformers:", transformers.__version__)
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
print("rdkit ok:", mol.GetNumAtoms(), "atoms")

import umap
import hdbscan

print("umap-learn:", umap.__version__, "| hdbscan:", hdbscan.__version__)

rng = np.random.RandomState(0)
X = np.vstack([rng.randn(1500, 32).astype("float32") + c * 6 for c in range(4)])

t0 = time.time()
emb_cpu = umap.UMAP(n_neighbors=15, n_components=16, random_state=42).fit_transform(X)
lab_cpu = hdbscan.HDBSCAN(min_cluster_size=50).fit_predict(emb_cpu)
n_cpu = len(set(lab_cpu.tolist()) - {-1})
print(f"CPU  umap+hdbscan ok in {time.time()-t0:.1f}s -> {n_cpu} clusters, noise={(lab_cpu==-1).sum()}")

try:
    import cuml
    print("cuml:", cuml.__version__)
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import HDBSCAN as cuHDBSCAN

    t0 = time.time()
    emb = cuUMAP(n_neighbors=15, n_components=16, random_state=42).fit_transform(X)
    emb_np = np.asarray(emb)
    lab = cuHDBSCAN(min_cluster_size=50).fit_predict(emb_np)
    lab_np = np.asarray(lab)
    n = len(set(lab_np.tolist()) - {-1})
    print(f"GPU  cuml umap+hdbscan ok in {time.time()-t0:.1f}s -> {n} clusters, noise={(lab_np==-1).sum()}")
except Exception as e:  # noqa: BLE001
    print("cuml FAILED:", type(e).__name__, e)

print("VERIFY COMPLETE")
