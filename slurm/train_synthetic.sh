#!/bin/bash
#SBATCH --job-name=rtf_syn
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_syn_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_syn_%j.err

set -euo pipefail

echo "=== RTF: Training with Synthetic Data ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

DATA=/fred/oz480/mcoughli/AppleCider/rtf/data_gp
SYNTH=/fred/oz480/mcoughli/AppleCider/rtf/data_synthetic
OUTDIR=/fred/oz480/mcoughli/AppleCider/rtf/runs
ANALYSIS=/fred/oz480/mcoughli/AppleCider/rtf/analysis

cd /fred/oz480/mcoughli/AppleCider/rtf/src

# AE + meta + GP + synthetic "other" class, random truncation, joint classification
echo ""
echo "========== Train: AE + meta + GP + synthetic (6 classes) =========="
python train.py \
    --data-dir $DATA \
    --synthetic-dir $SYNTH \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --use-gp \
    --gp-dim 114 \
    --latent-dims 64 \
    --random-truncate \
    --min-detections 3 \
    --cls-weight 0.5 \
    --num-classes 6 \
    --epochs 200 \
    --batch-size 128 \
    --num-workers 4

# Evaluate at each detection count
echo ""
echo "========== Evaluate at fixed detection counts =========="
python eval_early.py \
    --checkpoint $OUTDIR/ae_dim64_meta_gp_randtrunc_cls0.5/best_model.pt \
    --summary $OUTDIR/ae_dim64_meta_gp_randtrunc_cls0.5/summary.json \
    --data-dir $DATA \
    --output-dir $ANALYSIS/early_synthetic \
    --detection-counts 3 5 7 10 15 20 30 50 100

echo "Done: $(date)"
