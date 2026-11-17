#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Make both local HF models loadable in the yangsensen env.
#
# Problem 1: ChemBERTa-77M-MTR ships pytorch_model.bin. Since CVE-2025-32434
#            transformers refuses torch.load for .bin unless torch >= 2.6.
#            Fix: convert once to model.safetensors (safetensors path is exempt).
#            The original .bin is kept, nothing is deleted.
#
# Problem 2: MoLFormer's modeling_molformer.py (current HF revision) does
#            `from transformers.masking_utils import create_bidirectional_mask`,
#            which only exists in transformers 5.x (4.57.6 has masking_utils but
#            not that symbol). Fix: move to transformers 5.x.
#
# All writes inside /mnt/data/yangsensen.
# ---------------------------------------------------------------------------
set -uo pipefail

BASE=/mnt/data/yangsensen
PY=$BASE/envs/yangsensen/bin/python
CB=$BASE/embedding/chemberta2/code/ChemBERTa-77M-MTR

export PIP_CACHE_DIR=$BASE/pipcache
export TMPDIR=$BASE/tmp
export HF_HOME=$BASE/hf_cache
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"

echo "=== [1/3] convert ChemBERTa-77M-MTR .bin -> model.safetensors ==="
date
"$PY" - "$CB" <<'PYEOF'
import os
import sys

import torch
from safetensors.torch import save_file

d = sys.argv[1]
src = os.path.join(d, "pytorch_model.bin")
dst = os.path.join(d, "model.safetensors")
if os.path.exists(dst):
    print("already converted:", dst)
    sys.exit(0)
sd = torch.load(src, map_location="cpu", weights_only=True)
sd = {k: v.contiguous().clone() for k, v in sd.items()}
save_file(sd, dst, metadata={"format": "pt"})
print("saved", dst)
print("tensors:", len(sd), "params:", sum(v.numel() for v in sd.values()))
print("first keys:", list(sd)[:5])
PYEOF
if [ $? -ne 0 ]; then echo "SAFETENSORS CONVERSION FAILED"; fi

echo "=== [2/3] transformers 5.x ==="
date
"$PY" -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers==5.23.0" 2>&1 | tail -n 6

echo "=== [3/3] verify everything ==="
date
HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$BASE/verify_env2.py" 2>&1 | tail -n 50
echo "=== FIX MODELS DONE ==="
date
