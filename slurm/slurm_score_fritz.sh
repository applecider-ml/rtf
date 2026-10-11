#!/bin/bash
#SBATCH --job-name=rtf_score_fritz
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_score_fritz_%j.out

set -euo pipefail

echo "=== RTF Fritz Source Scoring ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "Assigned GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /work/hdd/bcrv/kmajithia/rtf/src

MODEL_PATH="/work/hdd/bcrv/kmajithia/rtf/runs/ae_1week_frozen/ae_dim128_meta_img/best_model.pt"
IF_PATH="/work/hdd/bcrv/kmajithia/rtf/runs/ae_1week_frozen/ae_dim128_meta_img/isolation_forest.joblib"
FRITZ_DIR="/work/hdd/bcrv/kmajithia/rtf/fritz_sources"
OUTPUT="/work/hdd/bcrv/kmajithia/rtf/runs/ae_1week_frozen/ae_dim128_meta_img/anomaly_scores.csv"

python score_fritz_sources.py \
    --model-path "$MODEL_PATH" \
    --if-path "$IF_PATH" \
    --fritz-dir "$FRITZ_DIR" \
    --output "$OUTPUT"

echo "Scoring complete: $(date)"
