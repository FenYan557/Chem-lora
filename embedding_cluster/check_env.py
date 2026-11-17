import importlib.util as u
mods = ["numpy", "pandas", "sklearn", "torch", "umap", "hdbscan", "rdkit", "scipy", "cuml", "cupy"]
for m in mods:
    print(m, "ok" if u.find_spec(m) else "MISSING", flush=True)
try:
    import torch
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)
except Exception as e:
    print("torch err", e, flush=True)
