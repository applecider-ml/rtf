#!/bin/bash
#SBATCH --job-name=tempo_fritz
#SBATCH --partition=cpu
#SBATCH --account=bcrv-delta-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/tempo_fritz_%j.out
#SBATCH --error=/work/hdd/bcrv/kmajithia/logs/tempo_fritz_%j.err

echo "=== TEMPO Tarball + Fritz Comparison: $(date) ==="

export TEMPO_ROOT="/projects/bcrv/ffontinelenunes/transformer/tempo"
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate

# Set your Fritz token here
export FRITZ_TOKEN="${FRITZ_TOKEN}"

ONNX="/work/hdd/bcrv/kmajithia/tempo/onnx/tempo_classifier.onnx"
SCRIPT="/work/hdd/bcrv/kmajithia/rtf/src/score_tarball_fritz.py"
OUTPUT="/work/hdd/bcrv/kmajithia/tempo/tarball_comparison"

# Use existing tarball or download latest
TARBALL="/work/hdd/bcrv/kmajithia/uw_archive/ztf_public_20260408.tar.gz"

echo "ONNX:    ${ONNX}"
echo "Tarball: ${TARBALL}"
echo "Output:  ${OUTPUT}"

mkdir -p "${OUTPUT}"

python "${SCRIPT}" \
    --tarball "${TARBALL}" \
    --onnx "${ONNX}" \
    --fritz-token "${FRITZ_TOKEN}" \
    --fritz-sample 500 \
    --max-alerts 50000 \
    --output-dir "${OUTPUT}"

echo ""
echo "=== Finished: $(date) ==="
ls -lh "${OUTPUT}/"
