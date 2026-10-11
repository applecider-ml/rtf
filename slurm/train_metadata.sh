#!/bin/bash
#SBATCH --job-name=rtf_meta
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_meta_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_meta_%j.err

set -euo pipefail

echo "=== RTF: AE with Alert Metadata ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

cd /fred/oz480/mcoughli/AppleCider/rtf/src

ALERT_DIR=/fred/oz480/mcoughli/data_ztf
SPLITS=/fred/oz480/mcoughli/AppleCider/photo_events/splits.csv
LABELS=/fred/oz480/mcoughli/AppleCider/photo_events
DATA=/fred/oz480/mcoughli/AppleCider/rtf/data
OUTDIR=/fred/oz480/mcoughli/AppleCider/rtf/runs

echo ""
echo "========== Step 1: Preprocess alerts → compact NPZ =========="
if [ ! -d "$DATA/train" ] || [ "$(ls $DATA/train/*.npz 2>/dev/null | wc -l)" -lt 1000 ]; then
    python preprocess_alerts.py \
        --alert-dir $ALERT_DIR \
        --splits $SPLITS \
        --labels-dir $LABELS \
        --output-dir $DATA \
        --horizon 100
else
    echo "  Preprocessed data already exists, skipping."
fi

echo ""
echo "========== Step 2: Train AE with metadata (37 channels) =========="
python train.py \
    --data-dir $DATA \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --latent-dims 8 32 64 128 256 512 \
    --epochs 200 \
    --batch-size 128 \
    --num-workers 4

echo ""
echo "========== Step 3: Linear probes =========="
python linear_probe.py \
    --runs-dir $OUTDIR \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis

echo ""
echo "========== Step 4: Visualizations =========="
pip install umap-learn -q 2>/dev/null || true
python visualize.py \
    --runs-dir $OUTDIR \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis/visualizations \
    --methods umap tsne \
    --comparison-dims 8 32 128 512

echo "Done: $(date)"
