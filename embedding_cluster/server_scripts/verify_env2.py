# -*- coding: utf-8 -*-
"""Full sanity check of the yangsensen env.

Every section is independent: one failure never hides the others.
Checks versions, CUDA, torch<->numpy ABI, CPU umap/hdbscan, GPU cuml
umap/hdbscan, RDKit fingerprints, and real fp16 loading of both local models.
"""
import importlib.metadata as md
import sys
import time
import traceback

PKGS = [
    "numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch",
    "transformers", "tokenizers", "huggingface_hub", "safetensors", "rdkit",
    "umap-learn", "hdbscan", "pynndescent", "numba", "llvmlite",
    "cuml-cu12", "cuvs-cu12", "rmm-cu12", "pylibraft-cu12", "treelite",
]
print("python:", sys.version.split()[0], sys.executable)
for p in PKGS:
    try:
        print("  %-16s %s" % (p, md.version(p)))
    except Exception:
        print("  %-16s NOT INSTALLED" % p)
sys.stdout.flush()

import numpy as np

FAILS = []


def section(name):
    print("\n=== %s ===" % name, flush=True)


# ---------------------------------------------------------------- torch/CUDA
section("torch + CUDA")
import torch

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("  gpu%d: %s %.1fGiB sm_%d%d" % (i, p.name, p.total_memory / 1024**3, p.major, p.minor))
x = torch.randn(2048, 2048, device="cuda:0", dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    y = x @ x
torch.cuda.synchronize()
dt = time.time() - t0
print("  fp16 matmul 2048^3 x10 in %.2fs (~%.1f GFLOP/s)" % (dt, 10 * 2 * 2048**3 / 1e9 / dt))

# torch<->numpy ABI (numpy 2.x vs torch built against numpy 1.x)
try:
    a = np.random.rand(16, 8).astype("float32")
    t = torch.from_numpy(a).to("cuda")
    back = (t * 2.0).cpu().numpy()
    ok = np.allclose(back, a * 2.0)
    print("  torch<->numpy interop:", "OK" if ok else "WRONG VALUES")
    if not ok:
        FAILS.append("torch<->numpy wrong values")
except Exception as e:
    print("  torch<->numpy interop FAILED:", type(e).__name__, e)
    FAILS.append("torch/numpy ABI: %s" % e)

# ---------------------------------------------------------------- CPU cluster
section("CPU umap-learn + hdbscan")
try:
    import hdbscan
    import umap

    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(1500, 32).astype("float32") + c * 6 for c in range(4)])
    t0 = time.time()
    e = umap.UMAP(n_neighbors=15, n_components=16, random_state=42).fit_transform(X)
    lab = hdbscan.HDBSCAN(min_cluster_size=50).fit_predict(e)
    n = len(set(lab.tolist()) - {-1})
    print("  OK in %.1fs -> %d clusters, noise=%d" % (time.time() - t0, n, int((lab == -1).sum())))
    if n < 2:
        FAILS.append("CPU clustering found <2 clusters on 4 obvious blobs")
except Exception as ex:
    print("  FAILED:", type(ex).__name__, ex)
    traceback.print_exc()
    FAILS.append("cpu umap/hdbscan: %s" % ex)

# ---------------------------------------------------------------- GPU cluster
section("GPU cuml UMAP + HDBSCAN")
try:
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    from cuml.manifold import UMAP as cuUMAP

    print("  cuml:", cuml.__version__)
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(20000, 64).astype("float32") + c * 8 for c in range(6)])
    t0 = time.time()
    emb = np.asarray(cuUMAP(n_neighbors=15, n_components=16, random_state=42).fit_transform(X))
    lab = np.asarray(cuHDBSCAN(min_cluster_size=100).fit_predict(emb))
    n = len(set(lab.tolist()) - {-1})
    print("  OK 20000x64 in %.1fs -> %d clusters, noise=%d, emb %s"
          % (time.time() - t0, n, int((lab == -1).sum()), emb.shape))
    if n < 3:
        FAILS.append("cuml clustering found <3 clusters on 6 obvious blobs")
except Exception as ex:
    print("  FAILED:", type(ex).__name__, ex)
    traceback.print_exc()
    FAILS.append("cuml: %s" % ex)

# ---------------------------------------------------------------- RDKit
section("RDKit fingerprints")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import ConvertToNumpyArray

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")), arr)
    print("  morgan r=2/2048 on-bits:", int(arr.sum()))
except Exception as ex:
    print("  FAILED:", type(ex).__name__, ex)
    FAILS.append("rdkit: %s" % ex)

# ---------------------------------------------------------------- HF models
section("local HF models (fp16 on cuda:0)")
MODELS = [
    ("MoLFormer-XL-both-10pct",
     "/mnt/data/yangsensen/embedding/molformer/code/MoLFormer-XL-both-10pct", True, 202),
    ("ChemBERTa-77M-MTR",
     "/mnt/data/yangsensen/embedding/chemberta2/code/ChemBERTa-77M-MTR", False, 256),
]
SMIS = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccc2c(c1)ccc1ccc3cccc4ccc2c34",
        "O=C(O)c1ccccc1N", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
sys.path.insert(0, "/mnt/data/yangsensen/embedding/molformer/code")
try:
    import molformer_compat

    molformer_compat.install()
except Exception as ex:
    print("  molformer_compat shim unavailable:", type(ex).__name__, ex)
try:
    from transformers import AutoModel, AutoTokenizer
    from transformers import __version__ as tv

    parts = tv.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    dtype_key = "dtype" if (major >= 5 or (major == 4 and minor >= 56)) else "torch_dtype"
    print("  transformers", tv, "-> from_pretrained kwarg:", dtype_key)
    for name, path, trc, maxlen in MODELS:
        try:
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trc)
            model = AutoModel.from_pretrained(path, trust_remote_code=trc,
                                              **{dtype_key: torch.float16}).cuda().eval()
            enc = tok(SMIS, padding=True, truncation=True, max_length=maxlen,
                      return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model(**enc).last_hidden_state[:, 0, :]
            vec = out.float().cpu().numpy()
            mem = torch.cuda.max_memory_allocated() / 1024**2
            print("  %s: emb %s %s |mean|=%.4f std=%.4f load+infer=%.1fs peak_gpu=%.0fMiB"
                  % (name, vec.shape, vec.dtype, float(np.abs(vec).mean()),
                     float(vec.std()), time.time() - t0, mem))
            if not np.isfinite(vec).all():
                FAILS.append("%s: non-finite embedding values" % name)
            del model, tok, enc, out
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as ex:
            print("  %s: FAILED %s: %s" % (name, type(ex).__name__, ex))
            traceback.print_exc()
            FAILS.append("%s: %s" % (name, ex))
except Exception as ex:
    print("  transformers import FAILED:", type(ex).__name__, ex)
    traceback.print_exc()
    FAILS.append("transformers: %s" % ex)

section("SUMMARY")
if FAILS:
    print("%d problem(s):" % len(FAILS))
    for f in FAILS:
        print("  -", f)
else:
    print("ALL CHECKS PASSED")
print("VERIFY2 COMPLETE")
