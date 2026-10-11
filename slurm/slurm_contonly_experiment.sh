#!/bin/bash
#SBATCH --job-name=rtf_contonly
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_contonly_%j.out

set -euo pipefail

echo "=== RTF Cont-Only Experiment (band_weight=0.0) ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

WORK="/work/hdd/bcrv/kmajithia/rtf"
EXPERIMENT="$WORK/runs/ae_contonly"
mkdir -p "$EXPERIMENT"

# ══════════════════════════════════════════════════════════════
# Stage 1: Copy shards to local NVMe + Train (band_weight=0.0)
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 1: Training cont-only model (3 epochs) ======"
GPFS_SHARDS="/work/hdd/bcrv/kmajithia/uw_tensors/shards"
LOCAL_SHARDS="/tmp/shards"
mkdir -p "$LOCAL_SHARDS"
echo "Copying shards to local NVMe..."
cp "$GPFS_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/
echo "  $(ls "$LOCAL_SHARDS"/shard_*.pt | wc -l) shards copied"

cd "$WORK/src"

python train.py \
    --mode ae \
    --use-metadata \
    --use-images \
    --data-dir "$LOCAL_SHARDS" \
    --output-dir "$EXPERIMENT" \
    --epochs 3 \
    --batch-size 256 \
    --latent-dim 128 \
    --d-model 256 \
    --num-workers 4 \
    --band-weight 0.0

rm -rf "$LOCAL_SHARDS"
echo "Stage 1 complete: $(date)"

# The trained model will be at $EXPERIMENT/ae_dim128_meta_img/best_model.pt
MODEL_DIR="$EXPERIMENT/ae_dim128_meta_img"

# ══════════════════════════════════════════════════════════════
# Stage 2: Extract reference embeddings + train Isolation Forest
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 2: Extracting reference embeddings ======"

# Copy shards back for reference extraction
mkdir -p "$LOCAL_SHARDS"
cp "$GPFS_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/

python extract_reference.py \
    --model-path "$MODEL_DIR/best_model.pt" \
    --shards-dir "$LOCAL_SHARDS" \
    --output-dir "$MODEL_DIR" \
    --batch-size 512

rm -rf "$LOCAL_SHARDS"
echo "Stage 2 complete: $(date)"

# ══════════════════════════════════════════════════════════════
# Stage 3: Score Fritz sources with the new model
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 3: Scoring Fritz sources ======"

python score_fritz_sources.py \
    --model-path "$MODEL_DIR/best_model.pt" \
    --if-path "$MODEL_DIR/isolation_forest.joblib" \
    --fritz-dir "$WORK/fritz_sources" \
    --output "$MODEL_DIR/anomaly_scores.csv"

echo "Stage 3 complete: $(date)"

# ══════════════════════════════════════════════════════════════
# Stage 4: Extract full embeddings + loss decomposition
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 4: Extracting Fritz embeddings ======"

python evaluate_embeddings.py \
    --model-path "$MODEL_DIR/best_model.pt" \
    --fritz-dir "$WORK/fritz_sources" \
    --output "$MODEL_DIR/fritz_embeddings.npz"

echo "Stage 4 complete: $(date)"

echo ""
echo "=== ALL STAGES COMPLETE: $(date) ==="
echo "  Model:      $MODEL_DIR/best_model.pt"
echo "  Ref embeds: $MODEL_DIR/reference_embeddings.npz"
echo "  IF:         $MODEL_DIR/isolation_forest.joblib"
echo "  Scores:     $MODEL_DIR/anomaly_scores.csv"
echo "  Embeddings: $MODEL_DIR/fritz_embeddings.npz"
