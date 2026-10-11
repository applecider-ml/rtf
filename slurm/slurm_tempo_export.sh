#!/bin/bash
#SBATCH --job-name=tempo_onnx
#SBATCH --partition=gpuA40x4
#SBATCH --account=bcrv-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/tempo_onnx_%j.out
#SBATCH --error=/work/hdd/bcrv/kmajithia/logs/tempo_onnx_%j.err

echo "=== TEMPO ONNX Export: $(date) ==="

# Paths
CHECKPOINT="/work/nvme/bcrv/ffontinelenunes/tempo_runs/sota_sota_push_earlyuq_2026_04_15_185642/d_train/p26_tv5_margin_refine_012_006/best_evidential.pt"
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/tempo/onnx"
SCRIPT="/work/hdd/bcrv/kmajithia/rtf/src/export_onnx_tempo.py"

# TEMPO codebase (for photometry_edl imports)
export TEMPO_ROOT="/projects/bcrv/ffontinelenunes/transformer/tempo"

# Activate our environment
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate

echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"

mkdir -p /work/hdd/bcrv/kmajithia/logs
mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Finished: $(date) ==="
echo "Output files:"
ls -lh "${OUTPUT_DIR}/"
