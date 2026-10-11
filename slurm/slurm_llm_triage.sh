#!/bin/bash
#SBATCH --job-name=rtf_llm_triage
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_llm_triage_%j.out

set -euo pipefail

echo "=== RTF LLM Triage ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1

WORK="/work/hdd/bcrv/kmajithia/rtf"
OUTPUT_DIR="$WORK/runs/ae_1week_frozen/ae_dim128_meta_img"

cd "$WORK/src"

# GROQ_API_KEY must be passed via environment or replaced below
# To pass it: export GROQ_API_KEY=gsk_XXXXX && sbatch slurm_llm_triage.sh
# Or replace YOUR_GROQ_KEY below with your actual key

python llm_triage_fritz.py \
    --scores-csv "$OUTPUT_DIR/anomaly_scores.csv" \
    --features "$WORK/fritz_features.parquet" \
    --manifest "$WORK/fritz_sources/fritz_manifest.csv" \
    --fritz-dir "$WORK/fritz_sources" \
    --groq-key "${GROQ_API_KEY}" \
    --top-n 30 \
    --min-detections 10 \
    --output-json "$OUTPUT_DIR/llm_triage_results.json" \
    --output-report "$OUTPUT_DIR/triage_report.md"

echo ""
echo "=== LLM Triage Complete: $(date) ==="
echo "  Results: $OUTPUT_DIR/llm_triage_results.json"
echo "  Report:  $OUTPUT_DIR/triage_report.md"
