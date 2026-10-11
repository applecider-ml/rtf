#!/bin/bash
#SBATCH --job-name=rtf_eval_emb
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_eval_emb_%j.out
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcrv-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

echo "=== RTF Embedding Evaluation ==="
echo "Start: $(date)"

# Environment
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate

# Paths
WORK="/work/hdd/bcrv/kmajithia/rtf"
MODEL_PATH="$WORK/runs/ae_1week_frozen/ae_dim128_meta_img/best_model.pt"
FRITZ_DIR="$WORK/fritz_sources"
OUTPUT="$WORK/runs/ae_1week_frozen/ae_dim128_meta_img/fritz_embeddings.npz"

cd "$WORK/src"

python evaluate_embeddings.py \
    --model-path "$MODEL_PATH" \
    --fritz-dir "$FRITZ_DIR" \
    --output "$OUTPUT"

echo ""
echo "=== Embedding Evaluation Complete: $(date) ==="
echo "  Output: $OUTPUT"
