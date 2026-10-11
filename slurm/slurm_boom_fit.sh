#!/bin/bash
#SBATCH --job-name=rtf_boom_fit
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_boom_fit_%j.out

set -euo pipefail

echo "=== RTF Phase 2: Rust Fitter + LLM Triage ==="
echo "Start: $(date)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$PATH"

WORK="/work/hdd/bcrv/kmajithia/rtf"
BOOM_ML="/work/hdd/bcrv/kmajithia/boom-fitting-ml"

# ── Step 1: Convert Fritz photometry to boom format ──
echo ""
echo "=== Step 1: Fritz → Boom format ==="
cd "$WORK/src"
python fritz_to_boom.py \
    --fritz-dir "$WORK/fritz_sources" \
    --output-dir "$WORK/boom_input"

# ── Step 2: Compile boom-fit-batch (if not already built) ──
echo ""
echo "=== Step 2: Compiling Rust fitter ==="
cd "$BOOM_ML"
if [ ! -f "target/release/boom-fit-batch" ]; then
    echo "  Building from source..."
    cargo build --release 2>&1
    echo "  Build complete."
else
    echo "  Binary already exists, skipping build."
fi

# ── Step 3: Run Rust fitter on all Fritz sources ──
echo ""
echo "=== Step 3: Running boom-fit-batch ==="
mkdir -p "$WORK/boom_output"
./target/release/boom-fit-batch \
    --input-dir "$WORK/boom_input" \
    --output-dir "$WORK/boom_output" \
    --sources "$WORK/boom_input/sources.csv" \
    --threads 16 \
    --max-detections 200

# ── Step 4: Build feature table ──
echo ""
echo "=== Step 4: Building feature table ==="

# Create empty cross-match parquet (we're skipping BOOM API catalog matching)
python -c "
import pandas as pd
pd.DataFrame({'obj_id': []}).set_index('obj_id').to_parquet('$WORK/boom_input/empty_xmatch.parquet')
print('Created empty xmatch parquet')
"

cd "$BOOM_ML/pipeline"
python build_features.py \
    --fitting-dir "$WORK/boom_output" \
    --xmatch "$WORK/boom_input/empty_xmatch.parquet" \
    --splits "$WORK/boom_input/sources.csv" \
    --npz-dir "$WORK/boom_input" \
    --output "$WORK/fritz_features.parquet"

echo ""
echo "=== All done: $(date) ==="
echo "  Boom output: $WORK/boom_output/"
echo "  Features: $WORK/fritz_features.parquet"
