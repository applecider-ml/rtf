#!/bin/bash
#SBATCH --job-name=compare_enc
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/compare_enc_%j.out

set -euo pipefail

echo "=== Encoder Comparison: RTF vs Perceiver IO vs AstroDistill ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ── Step 1: Define Dataset Paths ──
FRITZ_DIR="/work/hdd/bcrv/kmajithia/rtf/fritz_sources"
SCORES_CSV="/work/hdd/bcrv/kmajithia/rtf/runs/ae_color/ae_dim128_meta_img/anomaly_scores.csv"

echo "Using Fritz sources at: $FRITZ_DIR"
echo "Using scores at: $SCORES_CSV"

# ── Step 2: Run comparison ──
cd /work/hdd/bcrv/kmajithia/rtf/src

# Model paths
RTF_MODEL="/work/hdd/bcrv/kmajithia/rtf/runs/ae_dim128_meta_img/best_model.pt"
PERC_MODEL="/work/hdd/bcrv/kmajithia/rtf/runs/perceiver/best_model.pt"
ASTRO_MODEL="/work/hdd/bcrv/kmajithia/rtf/runs/astrodistill/best_model.pt"
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/rtf/runs/comparison"
mkdir -p "$OUTPUT_DIR"

# Build model flags (only include models that exist)
MODEL_FLAGS=""
if [ -f "$RTF_MODEL" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --rtf-model $RTF_MODEL"
    echo "  RTF model:    $RTF_MODEL"
else
    echo "  RTF model:    NOT FOUND (skipping)"
fi

if [ -f "$PERC_MODEL" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --perc-model $PERC_MODEL"
    echo "  Perceiver:    $PERC_MODEL"
else
    echo "  Perceiver:    NOT FOUND (skipping)"
fi

if [ -f "$ASTRO_MODEL" ]; then
    MODEL_FLAGS="$MODEL_FLAGS --astro-model $ASTRO_MODEL"
    echo "  AstroDistill: $ASTRO_MODEL"
else
    echo "  AstroDistill: NOT FOUND (skipping)"
fi

echo "Starting encoder comparison..."
python compare_encoders.py \
    --fritz-dir "$FRITZ_DIR" \
    --scores "$SCORES_CSV" \
    --output-dir "$OUTPUT_DIR" \
    $MODEL_FLAGS

echo "Comparison complete: $(date)"
