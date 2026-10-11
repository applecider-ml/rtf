#!/bin/bash
#SBATCH --job-name=rtf_gp
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_gp_%j.out

set -euo pipefail

echo "=== RTF GP Feature Experiment (band_weight=0.0, gp_weight=10.0) ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
source ~/.cargo/env
export PYTHONUNBUFFERED=1
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

WORK="/work/hdd/bcrv/kmajithia/rtf"
GPFS_SHARDS="/work/hdd/bcrv/kmajithia/uw_tensors/shards"
GP_SHARDS="/work/hdd/bcrv/kmajithia/uw_tensors/shards_gp"
EXPERIMENT="$WORK/runs/ae_gp10"
MODEL_DIR="$EXPERIMENT/ae_dim128_meta_img_gp"
LOCAL_SHARDS="/tmp/shards_gp"

mkdir -p "$EXPERIMENT" "$GP_SHARDS"

# ══════════════════════════════════════════════════════════════
# Stage 0: Compute GP features for training shards
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 0: Computing GP features for training shards ======"

# Remove old (possibly broken) GP shards and rebuild
rm -f "$GP_SHARDS"/shard_*.pt
cd "$WORK/src"
python compute_gp_shards.py \
    --input-dir "$GPFS_SHARDS" \
    --output-dir "$GP_SHARDS" \
    --workers 4

echo "  GP shards created: $(ls "$GP_SHARDS"/shard_*.pt | wc -l) files"
echo "Stage 0 complete: $(date)"

# ══════════════════════════════════════════════════════════════
# Stage 1: Copy GP shards to local NVMe + Train
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 1: Training with GP feature prediction ======"
mkdir -p "$LOCAL_SHARDS"
echo "Copying GP shards to local NVMe..."
cp "$GP_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/
echo "  $(ls "$LOCAL_SHARDS"/shard_*.pt | wc -l) shards copied"

cd "$WORK/src"

# Check if we have a checkpoint to resume from
RESUME_ARG=""
if [ -f "$MODEL_DIR/checkpoint.pt" ]; then
    echo "  Found checkpoint, will resume training!"
    RESUME_ARG="--resume-from $MODEL_DIR/checkpoint.pt"
fi

python train.py \
    --mode ae \
    --use-metadata \
    --use-images \
    --use-gp \
    --gp-dim 80 \
    --data-dir "$LOCAL_SHARDS" \
    --output-dir "$EXPERIMENT" \
    --epochs 3 \
    --batch-size 256 \
    --latent-dim 128 \
    --d-model 256 \
    --num-workers 4 \
    --band-weight 0.0 \
    --gp-weight 10.0 \
    $RESUME_ARG

rm -rf "$LOCAL_SHARDS"
echo "Stage 1 complete: $(date)"

# ══════════════════════════════════════════════════════════════
# Stage 2: Extract reference embeddings + train Isolation Forest
# ══════════════════════════════════════════════════════════════
echo ""
echo "====== STAGE 2: Extracting reference embeddings ======"

mkdir -p "$LOCAL_SHARDS"
cp "$GP_SHARDS"/shard_*.pt "$LOCAL_SHARDS"/

python extract_reference.py \
    --model-path "$MODEL_DIR/best_model.pt" \
    --shards-dir "$LOCAL_SHARDS" \
    --output-dir "$MODEL_DIR" \
    --batch-size 512

rm -rf "$LOCAL_SHARDS"
echo "Stage 2 complete: $(date)"

# ══════════════════════════════════════════════════════════════
# Stage 3: Score Fritz sources
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
echo "  GP Shards:  $GP_SHARDS"
echo "  Model:      $MODEL_DIR/best_model.pt"
echo "  Checkpoint: $MODEL_DIR/checkpoint.pt"
echo "  Scores:     $MODEL_DIR/anomaly_scores.csv"
echo "  Embeddings: $MODEL_DIR/fritz_embeddings.npz"
