#!/usr/bin/env bash
# 100k 中试 v2: 代表性(分布匹配)选点 —— 正比配额 + 簇内随机 + 噪声层按实测噪声率随机抽。
# 与 pilot_cluster.sh (旧: sqrt 配额 + 簇内 FPS + 噪声写死 5% 取最离群) 并存, 输出到
# 不同目录, 方便 QA 直接做前后对比。参数全部显式写出, 不依赖脚本默认值。
set -u
B=/mnt/data/yangsensen
P=$B/envs/yangsensen/bin/python
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=$B/tmp MPLCONFIGDIR=$B/tmp/mpl
cd $B/embedding
OUT=${OUT:-pilot_100k_v2}
T=${TARGET:-5000}
mkdir -p $OUT
FLAGS="--backend cuml --target $T --quota-power 1.0 --within random \
--noise-order random --noise-frac -1 --seed 42"
for m in "molformer:molformer/output" "chemberta2:chemberta2/output"; do
  name=${m%%:*}; dir=${m##*:}
  echo "=== $name start $(date +%T) ==="
  $P cluster_subset.py $FLAGS \
     --emb  $dir/${name}_emb_merged.npy \
     --meta $dir/${name}_smiles_merged.csv \
     --outdir $OUT/$name 2>&1 | tail -40
done
echo "=== morgan start $(date +%T) ==="
$P cluster_subset.py $FLAGS \
   --emb  morgan_fp/output/morgan_emb.npy \
   --meta morgan_fp/output/morgan_smiles.csv \
   --outdir $OUT/morgan 2>&1 | tail -40
echo "=== ALL DONE $(date +%T) ==="
