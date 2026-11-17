#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pin the yangsensen env to a known-good stack.
# Reason: pip default resolves transformers 5.x / pandas 3.x, but MoLFormer's
#         custom remote code (modeling_molformer.py) is written for
#         transformers 4.x, and pandas 3.x changes semantics we rely on.
# Writes only inside /mnt/data/yangsensen.
# ---------------------------------------------------------------------------
set -uo pipefail

BASE=/mnt/data/yangsensen
ENV_DIR=$BASE/envs/yangsensen
export CONDA_ENVS_DIRS=$BASE/envs
export CONDA_PKGS_DIRS=$BASE/pkgs
export PIP_CACHE_DIR=$BASE/pipcache
export TMPDIR=$BASE/tmp
export HF_HOME=$BASE/hf_cache

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR" || exit 1

echo "=== before ==="
python -m pip list 2>/dev/null | grep -iE '^(transformers|tokenizers|huggingface|pandas|numpy|torch|rdkit|umap|hdbscan|numba|llvmlite|scikit-learn|scipy|pyarrow|cuml|cuvs|rmm|pylibraft) '

echo "=== pin transformers 4.x + pandas 2.x + numpy 1.26 ==="
python -m pip install \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "transformers==4.51.3" \
  "tokenizers>=0.21,<0.22" \
  "huggingface_hub>=0.30,<1.0" \
  "safetensors>=0.4.3" \
  "pyarrow>=14" || { echo "PIN FAILED"; exit 1; }

echo "=== after ==="
python -m pip list 2>/dev/null | grep -iE '^(transformers|tokenizers|huggingface|pandas|numpy|torch|rdkit|umap|hdbscan|numba|llvmlite|scikit-learn|scipy|pyarrow|cuml|cuvs|rmm|pylibraft) '

echo "=== import check ==="
python - <<'PY'
import numpy, pandas, torch, transformers, tokenizers, huggingface_hub
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
print("rdkit", rdkit.__version__)
g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = g.GetFingerprint(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
print("morgan bits on:", fp.GetNumOnBits())
import umap, hdbscan, sklearn, scipy
print("umap", umap.__version__, "hdbscan", hdbscan.__version__, "sklearn", sklearn.__version__, "scipy", scipy.__version__)
try:
    import cuml
    print("cuml", cuml.__version__)
except Exception as e:
    print("cuml import failed:", type(e).__name__, e)
print("PIN VERIFY COMPLETE")
PY
