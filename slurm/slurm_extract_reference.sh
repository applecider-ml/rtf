#!/bin/bash
#SBATCH --job-name=rtf_extract_ref
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_extract_ref_%j.out

set -euo pipefail

echo "=== RTF Reference Embedding Extraction ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Copy shards to local NVMe ──
GPFS_SHARDS="/work/hdd/bcrv/kmajithia/uw_tensors/shards"
LOCAL_SHARDS="/tmp/shards"
mkdir -p "$LOCAL_SHARDS"
echo "Copying shards from GPFS to local NVMe..."
COPY_START=$SECONDS
cp "$GPFS_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/
COPY_TIME=$((SECONDS - COPY_START))
echo "  Copied $(ls "$LOCAL_SHARDS"/shard_*.pt | wc -l) shards in ${COPY_TIME}s"

# ── Step 2: Extract embeddings + train IF ──
cd /work/hdd/bcrv/kmajithia/rtf/src

MODEL_PATH="/work/hdd/bcrv/kmajithia/rtf/runs/ae_1week_frozen/ae_dim128_meta_img/best_model.pt"
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/rtf/runs/ae_1week_frozen/ae_dim128_meta_img"

python extract_reference.py \
    --model-path "$MODEL_PATH" \
    --shards-dir "$LOCAL_SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 512 \
    --num-workers 4

echo "Extraction complete: $(date)"

# Cleanup
rm -rf "$LOCAL_SHARDS"
echo "Cleaned up local NVMe."
