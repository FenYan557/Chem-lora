#!/usr/bin/env bash
# 双 3090 并行跑三种 embedding（GPU 各跑一半数据分片；Morgan 走 CPU 多进程）
#
# 用法（建议 nohup 挂后台，全量约几小时）:
#   cd /mnt/data/yangsensen/embedding
#   nohup bash run_all.sh > run_all.log 2>&1 &
#   LIMIT=2000 bash run_all.sh      # 小样验证：每个分片只跑前 2000 条
#   JOBS=16 bash run_all.sh         # 改 Morgan 的 CPU 进程数
#   SHARDS=4 bash run_all.sh        # 改 GPU 分片数（默认 2 = 双卡各一片）
# 跟踪进度: tail -f /mnt/data/yangsensen/embedding/*/output/*.log
set -u
B=/mnt/data/yangsensen
EMB=$B/embedding
PY=$B/envs/yangsensen/bin/python

# HF 必须离线（服务器连不上 huggingface.co），缓存/临时目录都落在数据盘
# （根分区 / 只剩 67G，写满会影响整机，别往 ~ 或 /tmp 写）
export HF_HOME=$B/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=$B/tmp MPLCONFIGDIR=$B/tmp/mpl
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

LIMIT=${LIMIT:-0}          # 每个 GPU 分片取前 N 条
JOBS=${JOBS:-32}
SHARDS=${SHARDS:-2}
LIM=""
MLIM=""
if [ "$LIMIT" != "0" ]; then
    LIM="--limit $LIMIT"
    # 分片是按 df.iloc[shard::SHARDS] 切的，所以 SHARDS 个分片各取前 LIMIT 条时，
    # 合起来正好覆盖 parquet 的前 LIMIT*SHARDS 行。Morgan 不分片、直接对全表
    # head(N)，必须传 LIMIT*SHARDS 才能和两个 GPU 模型覆盖同一批分子，
    # 否则 validate_emb.py 只能在交集（前 LIMIT 行）上比对。
    MLIM="--limit $((LIMIT * SHARDS))"
fi

echo "[run_all] start $(date '+%F %T')  LIMIT=$LIMIT  SHARDS=$SHARDS  JOBS=$JOBS"
echo "[run_all] python = $PY"

# 1) MoLFormer-XL (768d) -> GPU0 / GPU1 各跑一半
(cd "$EMB/molformer/code" && CUDA_VISIBLE_DEVICES=0 $PY embed_molformer.py --shard 0 --num-shards $SHARDS $LIM > ../output/molformer_shard0.log 2>&1) &
P1=$!
(cd "$EMB/molformer/code" && CUDA_VISIBLE_DEVICES=1 $PY embed_molformer.py --shard 1 --num-shards $SHARDS $LIM > ../output/molformer_shard1.log 2>&1) &
P2=$!

# 2) Morgan 指纹 (2048 bit, ECFP4) -> CPU（RDKit 多进程），可与 GPU 任务同时进行
(cd "$EMB/morgan_fp/code" && $PY embed_morgan.py --n-jobs $JOBS $MLIM > ../output/morgan.log 2>&1) &
P3=$!

wait $P1 $P2 $P3
echo "[run_all] molformer + morgan done $(date '+%F %T')"

# 3) ChemBERTa-2 (384d) -> 等 MoLFormer 结束后再上两卡，避免抢显存
(cd "$EMB/chemberta2/code" && CUDA_VISIBLE_DEVICES=0 $PY embed_chemberta2.py --shard 0 --num-shards $SHARDS $LIM > ../output/chemberta2_shard0.log 2>&1) &
P4=$!
(cd "$EMB/chemberta2/code" && CUDA_VISIBLE_DEVICES=1 $PY embed_chemberta2.py --shard 1 --num-shards $SHARDS $LIM > ../output/chemberta2_shard1.log 2>&1) &
P5=$!

wait $P4 $P5
echo "[run_all] ALL DONE $(date '+%F %T')"
