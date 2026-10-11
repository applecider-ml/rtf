#!/bin/bash
#SBATCH --job-name=rtf_onnx_export
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcrv-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_onnx_%j.out

# ── Export RTF model to ONNX for BOOM deployment ──
#
# Input:  best_model.pt (trained PyTorch checkpoint)
# Output: rtf_embed.onnx, rtf_recon.onnx
#
# Usage:
#   sbatch slurm_onnx_export.sh

set -euo pipefail

echo "=== RTF ONNX Export: $(date) ==="

# Paths
MODEL_DIR="/work/hdd/bcrv/kmajithia/rtf/runs/ae_color50/ae_dim128_meta_img"
MODEL_PATH="${MODEL_DIR}/best_model.pt"
OUTPUT_DIR="${MODEL_DIR}/onnx"
SRC_DIR="/work/hdd/bcrv/kmajithia/rtf/src"

# Environment
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate

# Install onnx + onnxruntime if missing
pip install --quiet onnx onnxruntime

echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

cd "${SRC_DIR}"
python export_onnx.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Export complete: $(date) ==="
echo "Files:"
ls -lh "${OUTPUT_DIR}"/*.onnx
echo ""
echo "Copy these to your BOOM repo:"
echo "  scp ${OUTPUT_DIR}/rtf_embed.onnx boom/data/models/"
echo "  scp ${OUTPUT_DIR}/rtf_recon.onnx boom/data/models/"
