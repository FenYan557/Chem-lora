#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Setup conda env "yangsensen" on gpu-server-3 for:
#   SMILES -> embeddings (MoLFormer / ChemBERTa2 / MorganFP) -> UMAP -> HDBSCAN
# ALL writes are confined to /mnt/data/yangsensen (env, pkgs, pip cache, tmp).
# Nothing under /, /opt, /home is modified.
# ---------------------------------------------------------------------------
set -uo pipefail

BASE=/mnt/data/yangsensen
ENV_DIR=$BASE/envs/yangsensen

export CONDA_ENVS_DIRS=$BASE/envs
export CONDA_PKGS_DIRS=$BASE/pkgs
export PIP_CACHE_DIR=$BASE/pipcache
export TMPDIR=$BASE/tmp
export HF_HOME=$BASE/hf_cache
export MPLCONFIGDIR=$BASE/tmp/mpl

mkdir -p "$BASE/envs" "$BASE/pkgs" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME" "$MPLCONFIGDIR"

source /opt/anaconda3/etc/profile.d/conda.sh

echo "=== [1/5] create conda env (python 3.11) ==="
date
if [ ! -x "$ENV_DIR/bin/python" ]; then
  conda create -p "$ENV_DIR" -c conda-forge python=3.11 -y || { echo "CONDA CREATE FAILED"; exit 1; }
else
  echo "env already exists, skip create"
fi
conda activate "$ENV_DIR" || { echo "ACTIVATE FAILED"; exit 1; }
echo "python: $(which python)  $(python -V 2>&1)"
echo "pip:    $(which pip)"

echo "=== [2/5] upgrade pip tooling ==="
date
python -m pip install -U pip setuptools wheel || { echo "PIP BOOTSTRAP FAILED"; exit 1; }

echo "=== [3/5] torch (cu121, matches driver CUDA 12.2) ==="
date
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 || { echo "TORCH INSTALL FAILED"; exit 1; }

echo "=== [4/5] chem + ML stack ==="
date
python -m pip install \
  "numpy<2" "pandas>=2" pyarrow \
  transformers safetensors tokenizers \
  rdkit \
  scikit-learn scipy joblib tqdm \
  umap-learn hdbscan pynndescent || { echo "ML STACK INSTALL FAILED"; exit 1; }

echo "=== [5/5] RAPIDS cuml (GPU UMAP + HDBSCAN) ==="
date
python -m pip install cuml-cu12 --extra-index-url https://pypi.nvidia.com || echo "CUML INSTALL FAILED (CPU fallback umap-learn/hdbscan still available)"

echo "=== VERIFY ==="
date
python "$BASE/verify_env.py"
echo "=== DONE ==="
date
echo "activate with:"
echo "  conda activate $ENV_DIR"
