#!/bin/bash
#SBATCH --job-name=rtf_early
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_early_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_early_%j.err

set -euo pipefail

echo "=== RTF: Early Classification (variable-length training + fixed-N evaluation) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

DATA=/fred/oz480/mcoughli/AppleCider/rtf/data_gp
OUTDIR=/fred/oz480/mcoughli/AppleCider/rtf/runs

cd /fred/oz480/mcoughli/AppleCider/rtf/src

# Step 1: Train a single model with random truncation
# Each epoch, each source is randomly truncated to [3, L] observations
echo ""
echo "========== Train: AE + meta + GP, random truncation [3, L] =========="
python train.py \
    --data-dir $DATA \
    --output-dir $OUTDIR \
    --mode ae \
    --use-metadata \
    --use-gp \
    --gp-dim 114 \
    --latent-dims 64 \
    --random-truncate \
    --min-detections 3 \
    --epochs 200 \
    --batch-size 128 \
    --num-workers 4

# Step 2: Evaluate at each detection count using the trained model
echo ""
echo "========== Evaluate at fixed detection counts =========="
python eval_early.py \
    --checkpoint $OUTDIR/ae_dim64_meta_gp_randtrunc/best_model.pt \
    --summary $OUTDIR/ae_dim64_meta_gp_randtrunc/summary.json \
    --data-dir $DATA \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis/early_classification \
    --detection-counts 3 5 7 10 15 20 30 50 100

echo "Done: $(date)"
