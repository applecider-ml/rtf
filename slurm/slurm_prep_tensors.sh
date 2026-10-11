#!/bin/bash
#SBATCH --job-name=prep_1week
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
cd /work/hdd/bcrv/kmajithia/rtf/src

DATES=("20260319" "20260320")

for DATE in "${DATES[@]}"; do
    echo "========================================="
    echo "Preprocessing $DATE into .npz tensors..."
    echo "========================================="
    OUT=/work/hdd/bcrv/kmajithia/uw_tensors/${DATE}
    mkdir -p $OUT
    
    python preprocess_alerts.py \
        --alert-dir /work/hdd/bcrv/kmajithia/uw_npy/${DATE} \
        --splits /work/hdd/bcrv/kmajithia/uw_npy/${DATE}/unlabelled_splits.csv \
        --labels-dir /work/hdd/bcrv/kmajithia/uw_npy/${DATE} \
        --output-dir $OUT \
        --workers 64
        
    echo "[SUCCESS] Finished $DATE. Instantly deleting raw .npy files to free up massive disk space..."
    rm -rf /work/hdd/bcrv/kmajithia/uw_npy/${DATE}
done

echo "All 4 days completely preprocessed into PyTorch .npz files!"
