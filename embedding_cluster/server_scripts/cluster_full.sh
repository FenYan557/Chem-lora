#!/usr/bin/env bash
# 全量流水线第二阶段: 合并 GPU 分片 -> 三套 embedding 依次聚类出子集。
# 必须在 run_all.sh 跑完之后启动 (本脚本不等待, 自己确认 log 里有 ALL DONE)。
#
# 用法:
#   nohup bash cluster_full.sh > cluster_full.log 2>&1 &
#   TARGET=100000 bash cluster_full.sh      # 换子集规模
#   FIT_N=1000000 bash cluster_full.sh      # 换 UMAP/HDBSCAN 拟合采样量
# 选点策略显式写成"代表性/分布匹配"那一套: 正比配额 + 簇内随机 + 噪声层按实测
# 噪声率随机抽。不依赖脚本默认值, 免得默认值以后被人改了这里跟着变。
set -u
B=/mnt/data/yangsensen
EMB=$B/embedding
PY=$B/envs/yangsensen/bin/python
export HF_HOME=$B/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=$B/tmp MPLCONFIGDIR=$B/tmp/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
cd "$EMB"
OUT=${OUT:-full_3.4m}
T=${TARGET:-50000}
FITN=${FIT_N:-500000}
mkdir -p "$OUT"

echo "[full] start $(date '+%F %T')  target=$T fit_n=$FITN out=$OUT"

for m in molformer chemberta2; do
  echo "=== merge $m $(date +%T) ==="
  $PY merge_shards.py --emb  "$m/output/${m}_emb_shard*of2.npy" \
                      --meta "$m/output/${m}_smiles_shard*of2.csv" \
      > "$OUT/merge_$m.log" 2>&1
  rc=$?
  tail -5 "$OUT/merge_$m.log"
  if [ $rc -ne 0 ]; then echo "[full] merge $m FAILED rc=$rc, 中止"; exit 1; fi
done

FLAGS="--backend cuml --target $T --fit-n $FITN --chunk 200000 --quota-power 1.0
--within random --noise-order random --noise-frac -1 --seed 42"

for m in "molformer:molformer/output" "chemberta2:chemberta2/output"; do
  name=${m%%:*}; dir=${m##*:}
  echo "=== cluster $name start $(date +%T) ==="
  $PY cluster_subset.py $FLAGS \
     --emb  "$dir/${name}_emb_merged.npy" \
     --meta "$dir/${name}_smiles_merged.csv" \
     --outdir "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "=== cluster $name rc=$? $(date +%T) ==="
  grep -E "pca|hdbscan|check|strength|select|done|Error|Traceback" "$OUT/$name.log" | tail -12
done

echo "=== cluster morgan start $(date +%T) ==="
$PY cluster_subset.py $FLAGS \
   --emb  morgan_fp/output/morgan_emb.npy \
   --meta morgan_fp/output/morgan_smiles.csv \
   --outdir "$OUT/morgan" > "$OUT/morgan.log" 2>&1
echo "=== cluster morgan rc=$? $(date +%T) ==="
grep -E "pca|hdbscan|check|strength|select|done|Error|Traceback" "$OUT/morgan.log" | tail -12

echo "[full] ALL DONE $(date '+%F %T')"
du -sh "$OUT"
free -g | head -2
