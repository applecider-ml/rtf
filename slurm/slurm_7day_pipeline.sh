#!/bin/bash
#SBATCH --job-name=ztf_rolling_7d
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/ztf_rolling_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

# First we must forcefully clean up the corrupted Day 4 from yesterday's crash
echo "[CLEANUP] Removing corrupted Day 4 extractions..."
rm -rf /work/hdd/bcrv/kmajithia/uw_npy/20260320
rm -rf /work/hdd/bcrv/kmajithia/uw_tensors/20260320

# We process the REMAINING 4 days (20th to 23rd) to combine with your
# existing 3 perfect days (17th to 19th) to mathematically hit the 7-day week!
DATES=("20260320" "20260321" "20260322" "20260323")

for DATE in "${DATES[@]}"; do
    echo "========================================="
    echo "STARTING ROLLING PIPELINE FOR $DATE"
    echo "========================================="
    
    # 1. Download directly from ZTF
    TARBALL="/work/hdd/bcrv/kmajithia/uw_archive/ztf_public_${DATE}.tar.gz"
    echo "[1/4] Downloading tarball for $DATE..."
    mkdir -p /work/hdd/bcrv/kmajithia/uw_archive/
    wget -qO "$TARBALL" "https://ztf.uw.edu/alerts/public/ztf_public_${DATE}.tar.gz" || { echo "Download failed"; exit 1; }
    
    # 2. Extract Avro to fast NVMe temporary storage
    AVRO_DIR="/tmp/uw_alerts_${DATE}"
    mkdir -p "$AVRO_DIR"
    echo "[2/4] Extracting tarball to NVMe..."
    tar -xzf "$TARBALL" -C "$AVRO_DIR"
    
    # 3. Convert Avro to NPY
    NPY_DIR="/work/hdd/bcrv/kmajithia/uw_npy/${DATE}"
    mkdir -p "$NPY_DIR"
    echo "[3/4] Running avro_to_npy.py..."
    source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
    export PYTHONPATH=/work/hdd/bcrv/kmajithia/boom-fitting-ml:${PYTHONPATH:-}
    python /work/hdd/bcrv/kmajithia/boom-fitting-ml/avro_to_npy.py \
        --avro-dir "$AVRO_DIR" \
        --output-dir "$NPY_DIR" \
        --workers 64
        
    # --- IMMEDIATE DISK CLEANUP PHASE 1 ---
    # This prevents the hard drive quota from EVER overflowing
    echo "[CLEANUP 1] Safely deleting $DATE tarball and NVMe files..."
    rm -f "$TARBALL"
    rm -rf "$AVRO_DIR"
    
    # 4. Convert NPY to NPZ PyTorch Tensors
    OUT="/work/hdd/bcrv/kmajithia/uw_tensors/${DATE}"
    mkdir -p "$OUT"
    echo "[4/4] Preprocessing into PyTorch NPZ tensors..."
    cd /work/hdd/bcrv/kmajithia/rtf/src
    python preprocess_alerts.py \
        --alert-dir "$NPY_DIR" \
        --splits "$NPY_DIR/unlabelled_splits.csv" \
        --labels-dir "$NPY_DIR" \
        --output-dir "$OUT" \
        --workers 64
        
    # --- IMMEDIATE DISK CLEANUP PHASE 2 ---
    echo "[CLEANUP 2] Safely deleting $DATE raw NPY files..."
    rm -rf "$NPY_DIR"
    
done

echo "========================================="
echo "ROLLING 7-DAY PIPELINE COMPLETELY FINISHED!"
echo "========================================="
