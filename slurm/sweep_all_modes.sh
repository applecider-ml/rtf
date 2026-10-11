#!/bin/bash
#SBATCH --job-name=comp_sweep
#SBATCH --partition=milan-c
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/comp_sweep_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/comp_sweep_%j.err

set -euo pipefail

echo "=== Alert Compression: AE vs VAE vs VQ-VAE ==="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

DATA=/fred/oz480/mcoughli/AppleCider/photo_events
OUTDIR=/fred/oz480/mcoughli/AppleCider/alert-compression/runs
cd /fred/oz480/mcoughli/AppleCider/alert-compression/src

DIMS="2 4 8 16 32 64 128 256 512 1024"

echo ""
echo "========== AE =========="
python train.py --data-dir $DATA --output-dir $OUTDIR --mode ae \
    --latent-dims $DIMS --epochs 200 --batch-size 128 --num-workers 4

echo ""
echo "========== VAE =========="
python train.py --data-dir $DATA --output-dir $OUTDIR --mode vae \
    --latent-dims $DIMS --epochs 200 --batch-size 128 --beta 1.0 --num-workers 4

echo ""
echo "========== VQ-VAE =========="
python train.py --data-dir $DATA --output-dir $OUTDIR --mode vqvae \
    --latent-dims $DIMS --epochs 200 --batch-size 128 --num-codes 512 --num-workers 4

echo ""
echo "========== Linear Probes =========="
python linear_probe.py --runs-dir $OUTDIR \
    --output-dir /fred/oz480/mcoughli/AppleCider/alert-compression/analysis

echo "Done: $(date)"
