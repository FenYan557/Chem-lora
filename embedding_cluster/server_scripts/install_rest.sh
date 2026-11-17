#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Step 4 + 5 of the yangsensen env install, using a FAST mirror.
#   * pypi.org (files.pythonhosted.org) measured ~29 kB/s on this box -> unusable
#   * tuna mirror measured ~8 MB/s
# Versions are pinned to a known-good stack:
#   transformers 4.51.x  (MoLFormer's custom remote code targets 4.x, not 5.x)
#   pandas 2.2.x, numpy 1.26.x
# Writes only inside /mnt/data/yangsensen.
# ---------------------------------------------------------------------------
set -uo pipefail

BASE=/mnt/data/yangsensen
ENV_DIR=$BASE/envs/yangsensen
INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

export CONDA_ENVS_DIRS=$BASE/envs
export CONDA_PKGS_DIRS=$BASE/pkgs
export PIP_CACHE_DIR=$BASE/pipcache
export TMPDIR=$BASE/tmp
export HF_HOME=$BASE/hf_cache
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR" || { echo "ACTIVATE FAILED"; exit 1; }

echo "=== [4/5] chem + ML stack (pinned, via tuna) ==="
date
python -m pip install -i "$INDEX" \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "pyarrow>=14" \
  "transformers==4.51.3" \
  "tokenizers>=0.21,<0.22" \
  "huggingface_hub>=0.30,<1.0" \
  "safetensors>=0.4.3" \
  "rdkit" \
  "scikit-learn>=1.3" "scipy>=1.11" "joblib" "tqdm" \
  "umap-learn" "hdbscan" "pynndescent" \
  || { echo "ML STACK INSTALL FAILED"; exit 1; }
echo "--- step4 done ---"
date

echo "=== [5/5] RAPIDS cuml-cu12 (GPU UMAP + HDBSCAN) ==="
date
python -m pip install -i "$INDEX" --extra-index-url https://pypi.nvidia.com cuml-cu12 \
  || echo "CUML INSTALL FAILED -> CPU fallback (umap-learn/hdbscan) still available"
echo "--- step5 done ---"
date

echo "=== INSTALLED VERSIONS ==="
python -m pip list 2>/dev/null | grep -iE '^(numpy|pandas|pyarrow|torch|transformers|tokenizers|huggingface|safetensors|rdkit|scikit-learn|scipy|numba|llvmlite|umap|hdbscan|pynndescent|cuml|cuvs|rmm|pylibraft|treelite|joblib|tqdm) '

echo "=== VERIFY ==="
date
python "$BASE/verify_env.py"
echo "=== DONE ==="
date
