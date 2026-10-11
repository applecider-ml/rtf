#!/bin/bash
#SBATCH --job-name=fritz_cmp
#SBATCH --partition=cpu
#SBATCH --account=bcrv-delta-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/fritz_cmp_%j.out
#SBATCH --error=/work/hdd/bcrv/kmajithia/logs/fritz_cmp_%j.err

echo "=== Fritz Full Comparison: $(date) ==="

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate

SCRIPT="/work/hdd/bcrv/kmajithia/rtf/src/fritz_compare_only.py"
PREDS="/work/hdd/bcrv/kmajithia/tempo/tarball_comparison/tempo_tarball_predictions.csv"
OUTPUT="/work/hdd/bcrv/kmajithia/tempo/tarball_comparison"

echo "Predictions: ${PREDS}"
echo "Output:      ${OUTPUT}"
echo "Objects:     $(wc -l < ${PREDS}) lines"

python "${SCRIPT}" \
    --predictions "${PREDS}" \
    --fritz-token "${FRITZ_TOKEN}" \
    --output-dir "${OUTPUT}"

echo ""
echo "=== Finished: $(date) ==="
ls -lh "${OUTPUT}/"
