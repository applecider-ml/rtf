#!/bin/bash
#SBATCH --job-name=rtf_full
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_full_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_full_%j.err

set -euo pipefail

echo "=== RTF: Full multimodal AE (photometry + metadata + images + GP) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

DATA=/fred/oz480/mcoughli/AppleCider/rtf/data_gp
OUTDIR=/fred/oz480/mcoughli/AppleCider/rtf/runs

cd /fred/oz480/mcoughli/AppleCider/rtf/src

echo ""
echo "========== Check preprocessed data =========="
if [ ! -d "$DATA/train" ] || [ "$(ls $DATA/train/*.npz 2>/dev/null | wc -l)" -lt 1000 ]; then
    echo "ERROR: Preprocessed data not found in $DATA."
    exit 1
fi
echo "  Found $(ls $DATA/train/*.npz | wc -l) train files"

echo ""
echo "========== AE + metadata + images + GP =========="
python train.py \
    --data-dir $DATA \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --use-images \
    --use-gp \
    --gp-dim 114 \
    --latent-dims 32 64 128 256 \
    --epochs 200 \
    --batch-size 64 \
    --num-workers 4

echo ""
echo "========== AE + metadata + GP (no images, faster) =========="
python train.py \
    --data-dir $DATA \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --use-gp \
    --gp-dim 114 \
    --latent-dims 32 64 128 256 \
    --epochs 200 \
    --batch-size 128 \
    --num-workers 4

echo ""
echo "========== Linear probes =========="
python linear_probe.py \
    --runs-dir $OUTDIR \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis

echo "Done: $(date)"
