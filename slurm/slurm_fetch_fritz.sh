#!/bin/bash
#SBATCH --job-name=fritz_fetch
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/fritz_fetch_%j.out

set -euo pipefail

echo "=== Fritz Source Fetcher ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1

WORK="/work/hdd/bcrv/kmajithia/rtf"
OUTPUT_DIR="$WORK/fritz_sources"

# ── SET YOUR FRITZ TOKEN HERE ──
FRITZ_TOKEN="1f113d94-89b7-4355-a049-84473a9286f4"

mkdir -p "$OUTPUT_DIR"

echo "Output dir: $OUTPUT_DIR"
echo "Groups: 53 (SLSNe), 85 (Fast Transients), 1876 (LIONS)"

cd "$WORK/src"

python fetch_fritz_sources.py \
    --token "$FRITZ_TOKEN" \
    --group-ids "53,85,1876" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Fritz fetch complete: $(date)"
echo "Results at: $OUTPUT_DIR"
