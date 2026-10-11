#!/bin/bash
#SBATCH --job-name=rtf_dino_128
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_dinov2_128d_%j.out

set -euo pipefail

echo "=== RTF 128-D + DINOv2 ViT-S/14 (~22M frozen) ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Pre-download DINOv2 weights if needed ──
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

OUTPUT_DIR="/work/hdd/bcrv/kmajithia/rtf/runs/rtf_dinov2_128d"
mkdir -p "$OUTPUT_DIR"

echo "Starting RTF 128-D + DINOv2 training on 1x GPU..."
echo "NOTE: DINOv2 ViT-S/14 has ~22M frozen params. If OOM, reduce --batch-size to 128."
python train.py \
    --mode ae \
    --use-metadata \
    --use-images \
    --image-backbone dinov2 \
    --data-dir "$LOCAL_SHARDS" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 25 \
    --batch-size 256 \
    --latent-dims 128 \
    --d-model 256 \
    --num-workers 4

echo "Training complete: $(date)"

# Cleanup
rm -rf "$LOCAL_SHARDS"
echo "Cleaned up local NVMe shards."
