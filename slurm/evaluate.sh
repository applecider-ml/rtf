#!/bin/bash
#SBATCH --job-name=comp_eval
#SBATCH --partition=milan-c
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/comp_eval_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/comp_eval_%j.err

set -euo pipefail

echo "=== Alert Compression: Physical Evaluation + MLP Decoders ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

RUNS=/fred/oz480/mcoughli/AppleCider/alert-compression/runs
DATA=/fred/oz480/mcoughli/AppleCider/photo_events
ANALYSIS=/fred/oz480/mcoughli/AppleCider/alert-compression/analysis

cd /fred/oz480/mcoughli/AppleCider/alert-compression/src

echo ""
echo "========== Physical Metrics + Light Curve Plots =========="
python evaluate_physical.py \
    --runs-dir $RUNS \
    --data-dir $DATA \
    --output-dir $ANALYSIS/physical \
    --plot-per-class 5

echo ""
echo "========== MLP Decoder Classifiers =========="
python mlp_decoder.py \
    --runs-dir $RUNS \
    --output-dir $ANALYSIS/decoders

echo "Done: $(date)"
