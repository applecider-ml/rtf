#!/bin/bash
#SBATCH --job-name=astro_dino
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/astrodistill_dinov2_%j.out

set -euo pipefail

echo "=== AstroDistill + DINOv2 ViT-S/14 Training ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Pre-download DINOv2 weights on login node if needed ──
# DINOv2 weights are downloaded via torch.hub on first use (~85MB).
# If compute nodes lack internet, run this on a login node first:
#   python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)"
echo "Verifying torch.hub is available..."
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"

# ── Step 1: Copy shards from GPFS to local NVMe ──
GPFS_SHARDS="/work/hdd/bcrv/kmajithia/uw_tensors/shards"
LOCAL_SHARDS="/tmp/shards"
mkdir -p "$LOCAL_SHARDS"
echo "Copying shard files from GPFS to local NVMe..."
echo "  Source: $GPFS_SHARDS"
echo "  Dest:   $LOCAL_SHARDS"
COPY_START=$SECONDS
cp "$GPFS_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/
COPY_TIME=$((SECONDS - COPY_START))
N_SHARDS=$(ls "$LOCAL_SHARDS"/shard_*.pt | wc -l)
echo "  Copied $N_SHARDS shards in ${COPY_TIME}s"
echo "  Local NVMe usage: $(du -sh $LOCAL_SHARDS | cut -f1)"

# ── Step 2: Train ──
cd /work/hdd/bcrv/kmajithia/rtf/src

OUTPUT_DIR="/work/hdd/bcrv/kmajithia/rtf/runs/astrodistill_dinov2"
mkdir -p "$OUTPUT_DIR"

# Check for existing checkpoint to resume from
RESUME_FLAG=""
if [ -f "$OUTPUT_DIR/checkpoint.pt" ]; then
    echo "Found existing checkpoint, resuming..."
    RESUME_FLAG="--resume-from $OUTPUT_DIR/checkpoint.pt"
fi

echo "Starting AstroDistill + DINOv2 training on 1x A100..."
echo "NOTE: DINOv2 ViT-S/14 has ~22M frozen params. If OOM, reduce --batch-size to 128 and increase --grad-accum to 4."
python train_astrodistill.py \
    --data-dir "$LOCAL_SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 25 \
    --batch-size 256 \
    --grad-accum 2 \
    --lr 1e-4 \
    --d-model 256 \
    --embed-dim 128 \
    --n-layers 4 \
    --mask-ratio 0.5 \
    --ema-decay 0.999 \
    --ema-end 0.9999 \
    --top-k-layers 2 \
    --warmup-epochs 3 \
    --num-workers 4 \
    --image-backbone dinov2 \
    $RESUME_FLAG

echo "Training complete: $(date)"

# Cleanup
rm -rf "$LOCAL_SHARDS"
echo "Cleaned up local NVMe shards."
