#!/bin/bash
#SBATCH --job-name=rtf_pack_shards
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_pack_shards_%j.out

set -euo pipefail

echo "=== RTF Dataset Sharding ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1
cd /work/hdd/bcrv/kmajithia/rtf/src

# All 7 days of preprocessed tensor data
INPUT_DIRS="/work/hdd/bcrv/kmajithia/uw_tensors/20260317,/work/hdd/bcrv/kmajithia/uw_tensors/20260318,/work/hdd/bcrv/kmajithia/uw_tensors/20260319,/work/hdd/bcrv/kmajithia/uw_tensors/20260320,/work/hdd/bcrv/kmajithia/uw_tensors/20260321,/work/hdd/bcrv/kmajithia/uw_tensors/20260322,/work/hdd/bcrv/kmajithia/uw_tensors/20260323"

# Output: large .pt shard files
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/uw_tensors/shards"

python pack_shards.py \
    --input-dirs "$INPUT_DIRS" \
    --output-dir "$OUTPUT_DIR" \
    --shard-size 20000 \
    --workers 16

echo "Sharding complete: $(date)"
