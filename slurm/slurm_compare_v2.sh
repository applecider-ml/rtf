#!/bin/bash
#SBATCH --job-name=compare_v2
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/compare_v2_%j.out

set -euo pipefail

echo "=== 16-Model Encoder Comparison v3 (Phase 1 + Phase 2 + Phase 3) ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
echo "Running on CPU (no GPU requested)"

WORK="/work/hdd/bcrv/kmajithia/rtf"
FRITZ_DIR="$WORK/fritz_sources"
SCORES_CSV="$WORK/runs/ae_color/ae_dim128_meta_img/anomaly_scores.csv"
RUNS_DIR="$WORK/runs"
OUTPUT_DIR="$WORK/runs/comparison_v3"

mkdir -p "$OUTPUT_DIR"

echo "Fritz dir:  $FRITZ_DIR"
echo "Scores:     $SCORES_CSV"
echo "Runs dir:   $RUNS_DIR"
echo "Output dir: $OUTPUT_DIR"

echo ""
echo "Available model checkpoints:"
for d in ae_dim128_meta_img perceiver perceiver_32d perceiver_64d \
         astrodistill astrodistill_32d astrodistill_64d \
         astrodistill_zoobot astrodistill_dinov2 astrodistill_zoobot_64d \
         rtf_zoobot_large_64d rtf_zoobot_large_128d \
         ae_dim64 \
         rtf_zoobot_pico_64d/ae_dim64_meta_img \
         rtf_zoobot_pico_128d/ae_dim128_meta_img \
         rtf_dinov2_128d/ae_dim128_meta_img; do
    if [ -f "$RUNS_DIR/$d/best_model.pt" ]; then
        echo "  ✓ $d"
    else
        echo "  ✗ $d (missing)"
    fi
done

cd "$WORK/src"

python compare_encoders_v2.py \
    --fritz-dir "$FRITZ_DIR" \
    --scores "$SCORES_CSV" \
    --runs-dir "$RUNS_DIR" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Comparison complete: $(date)"
echo "Results at: $OUTPUT_DIR"
