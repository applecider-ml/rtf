#!/bin/bash
#SBATCH --job-name=tempo_val
#SBATCH --partition=cpu
#SBATCH --account=bcrv-delta-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/tempo_val_%j.out
#SBATCH --error=/work/hdd/bcrv/kmajithia/logs/tempo_val_%j.err

echo "=== TEMPO Validation: $(date) ==="

# TEMPO codebase (for photometry_edl imports)
export TEMPO_ROOT="/projects/bcrv/ffontinelenunes/transformer/tempo"

# Activate environment
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate

ONNX="/work/hdd/bcrv/kmajithia/tempo/onnx/tempo_classifier.onnx"
DATA="/projects/bcrv/ffontinelenunes/transformer/tempo/data/photo_events"
OUTPUT="/work/hdd/bcrv/kmajithia/tempo/validation"
SCRIPT="/work/hdd/bcrv/kmajithia/rtf/src/validate_tempo.py"

echo "ONNX:   ${ONNX}"
echo "Data:   ${DATA}"
echo "Output: ${OUTPUT}"

mkdir -p "${OUTPUT}"

python "${SCRIPT}" \
    --onnx "${ONNX}" \
    --data-dir "${DATA}" \
    --output-dir "${OUTPUT}"

echo ""
echo "=== Finished: $(date) ==="
ls -lh "${OUTPUT}/"
