#!/bin/bash
#SBATCH --job-name=rtf_img
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_img_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_img_%j.err

set -euo pipefail

echo "=== RTF: Multimodal AE (photometry + metadata + images) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

ALERT_DIR=/fred/oz480/mcoughli/data_ztf
SPLITS=/fred/oz480/mcoughli/AppleCider/photo_events/splits.csv
LABELS=/fred/oz480/mcoughli/AppleCider/photo_events
DATA=/fred/oz480/mcoughli/AppleCider/rtf/data_img
OUTDIR=/fred/oz480/mcoughli/AppleCider/rtf/runs

cd /fred/oz480/mcoughli/AppleCider/rtf/src

echo ""
echo "========== Step 1: Check preprocessed data =========="
if [ ! -d "$DATA/train" ] || [ "$(ls $DATA/train/*.npz 2>/dev/null | wc -l)" -lt 1000 ]; then
    echo "ERROR: Preprocessed data not found in $DATA."
    echo "Run preprocessing on the head node first:"
    echo "  python src/preprocess_alerts.py --alert-dir $ALERT_DIR --splits $SPLITS --labels-dir $LABELS --output-dir $DATA --horizon 100"
    exit 1
fi
echo "  Found $(ls $DATA/train/*.npz | wc -l) train files"

echo ""
echo "========== Step 2: Train AE with metadata + images =========="
python train.py \
    --data-dir $DATA \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --use-images \
    --latent-dims 32 64 128 256 \
    --epochs 200 \
    --batch-size 64 \
    --num-workers 4

echo ""
echo "========== Step 3: Linear probes =========="
python linear_probe.py \
    --runs-dir $OUTDIR \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis

echo ""
echo "========== Step 4: Visualizations =========="
python visualize.py \
    --runs-dir $OUTDIR \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis/visualizations \
    --models ae_dim64_meta_img ae_dim256_meta_img \
    --methods umap tsne \
    --comparison-only

echo "Done: $(date)"
