#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Final env fix: go back to transformers 4.57.6 (last 4.x, works with the
# installed torch 2.5.1+cu121) and rely on molformer_compat.py for the single
# transformers-5-only symbol MoLFormer needs. Then verify everything.
# All writes inside /mnt/data/yangsensen.
# ---------------------------------------------------------------------------
set -uo pipefail

BASE=/mnt/data/yangsensen
PY=$BASE/envs/yangsensen/bin/python
export PIP_CACHE_DIR=$BASE/pipcache
export TMPDIR=$BASE/tmp
export HF_HOME=$BASE/hf_cache

echo "=== [1/3] transformers -> 4.57.6 ==="
date
"$PY" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "transformers==4.57.6" "tokenizers>=0.22,<0.23" "huggingface_hub>=0.34,<2.0" 2>&1 | tail -n 6

echo "=== [2/3] rdkit API probe (embed_morgan.py compatibility) ==="
"$PY" - <<'PYEOF'
from rdkit.Chem import AllChem, rdFingerprintGenerator
print("legacy GetMorganFingerprintAsBitVect present:", hasattr(AllChem, "GetMorganFingerprintAsBitVect"))
print("rdFingerprintGenerator present:", hasattr(rdFingerprintGenerator, "GetMorganGenerator"))
PYEOF

echo "=== [3/3] full verify (with MoLFormer compat shim) ==="
date
HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$BASE/verify_env2.py" 2>&1 | tail -n 55
echo "=== REVERT DONE ==="
date
