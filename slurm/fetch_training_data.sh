#!/bin/bash
# Description: Downloads 7 days of raw ZTF public tarballs and unpacks them
# into PyTorch-ready .npz tensors without running any inference.

DATES=("20260317" "20260318" "20260319" "20260320" "20260321" "20260322" "20260323")
TARGET_DIR="/work/hdd/bcrv/kmajithia/uw_archive"
mkdir -p "$TARGET_DIR"

echo "=========================================================="
echo "Phase 1: Downloading & Unpacking 1-Week Training Set"
echo "=========================================================="

for DATE in "${DATES[@]}"; do
    TARBALL_PATH="$TARGET_DIR/ztf_public_${DATE}.tar.gz"
    
    if [ ! -f "$TARBALL_PATH" ]; then
        echo "[$(date)] Downloading ztf_public_${DATE}.tar.gz..."
        wget -q -O "$TARBALL_PATH" "https://ztf.uw.edu/alerts/public/ztf_public_${DATE}.tar.gz"
    else
        echo "[$(date)] Tarball for $DATE already exists. Skipping download."
    fi
    
    # ONLY trigger the ingest/unpack stage. --wait ensures we process sequentially.
    echo "[$(date)] Extracting $DATE into .npz tensors..."
    sbatch --wait /work/hdd/bcrv/kmajithia/cyoa/ingest/slurm_unpack.sh "$DATE"
done

echo "=========================================================="
echo "Phase 1 Complete! Tensors are ready for PyTorch."
echo "=========================================================="
