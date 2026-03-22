#!/bin/bash
#SBATCH --job-name=rtf_gp_prep
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_gp_prep_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_gp_prep_%j.err

set -euo pipefail

echo "=== RTF: Preprocess alerts with images + GP features (parallel) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1

cd /fred/oz480/mcoughli/AppleCider/rtf/src

python preprocess_alerts.py \
    --alert-dir /fred/oz480/mcoughli/data_ztf \
    --splits /fred/oz480/mcoughli/AppleCider/photo_events/splits.csv \
    --labels-dir /fred/oz480/mcoughli/AppleCider/photo_events \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/data_gp \
    --horizon 100 \
    --fit-gp \
    --workers 32

echo "Done: $(date)"
