#!/bin/bash
#SBATCH --job-name=rtf_train_1week
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_train_1week_%j.out

set -euo pipefail

echo "=== 1-Week RTF Base Autoencoder Training (Sharded) ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Copy shards from slow GPFS to blazing-fast local NVMe ──
GPFS_SHARDS="/work/hdd/bcrv/kmajithia/uw_tensors/shards"
LOCAL_SHARDS="/tmp/shards"
mkdir -p "$LOCAL_SHARDS"
echo "Copying 88 shard files from GPFS to local NVMe..."
echo "  Source: $GPFS_SHARDS"
echo "  Dest:   $LOCAL_SHARDS"
COPY_START=$SECONDS
cp "$GPFS_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/
COPY_TIME=$((SECONDS - COPY_START))
N_SHARDS=$(ls "$LOCAL_SHARDS"/shard_*.pt | wc -l)
echo "  Copied $N_SHARDS shards in ${COPY_TIME}s"
echo "  Local NVMe usage: $(du -sh $LOCAL_SHARDS | cut -f1)"

# ── Step 2: Train from local NVMe ──
cd /work/hdd/bcrv/kmajithia/rtf/src

RESUME_MODEL="/work/hdd/bcrv/kmajithia/rtf/runs/ae_dim128_meta_img/best_model.pt"
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/rtf/runs/ae_1week_frozen"
mkdir -p "$OUTPUT_DIR"

echo "Starting training on 1x GPU from local NVMe shards..."
python train.py \
    --mode ae \
    --use-metadata \
    --use-images \
    --data-dir "$LOCAL_SHARDS" \
    --resume-from "$RESUME_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 25 \
    --batch-size 256 \
    --latent-dim 128 \
    --d-model 256 \
    --num-workers 4

echo "Training complete: $(date)"

# Cleanup
rm -rf "$LOCAL_SHARDS"
echo "Cleaned up local NVMe shards."
