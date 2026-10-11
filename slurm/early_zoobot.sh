#!/bin/bash
#SBATCH --job-name=rtf_zoo
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_zoo_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_zoo_%j.err

set -euo pipefail

echo "=== RTF: Zoobot Early Classification ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1
export HUGGINGFACE_HUB_CACHE=/fred/oz480/mcoughli/model_cache
export HF_HOME=/fred/oz480/mcoughli/model_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA=/fred/oz480/mcoughli/AppleCider/rtf/data_gp
OUTDIR=/fred/oz480/mcoughli/AppleCider/rtf/runs
ANALYSIS=/fred/oz480/mcoughli/AppleCider/rtf/analysis

cd /fred/oz480/mcoughli/AppleCider/rtf/src

# Train with Zoobot image backbone + meta + GP, random truncation
echo ""
echo "========== Train: AE + meta + Zoobot + GP, random truncation =========="
python train.py \
    --data-dir $DATA \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --use-images \
    --image-backbone zoobot \
    --use-gp \
    --gp-dim 114 \
    --latent-dims 64 \
    --random-truncate \
    --min-detections 3 \
    --epochs 200 \
    --batch-size 16 \
    --num-workers 4

# Evaluate at each detection count
echo ""
echo "========== Evaluate Zoobot model at fixed detection counts =========="
python eval_early.py \
    --checkpoint $OUTDIR/ae_dim64_meta_img_gp_randtrunc/best_model.pt \
    --summary $OUTDIR/ae_dim64_meta_img_gp_randtrunc/summary.json \
    --data-dir $DATA \
    --output-dir $ANALYSIS/early_zoobot \
    --detection-counts 3 5 7 10 15 20 30 50 100

echo "Done: $(date)"
