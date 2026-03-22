#!/bin/bash
#SBATCH --job-name=rtf_eval
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fred/oz480/mcoughli/AppleCider/logs/rtf_eval_%j.out
#SBATCH --error=/fred/oz480/mcoughli/AppleCider/logs/rtf_eval_%j.err

set -euo pipefail

source /fred/oz480/mcoughli/envs/applecider-xgb/bin/activate
export PYTHONUNBUFFERED=1
export HUGGINGFACE_HUB_CACHE=/fred/oz480/mcoughli/model_cache
export HF_HOME=/fred/oz480/mcoughli/model_cache
export HF_HUB_OFFLINE=1

cd /fred/oz480/mcoughli/AppleCider/rtf/src

echo "=== Zoobot Early Classification Eval ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

python eval_early.py \
    --checkpoint /fred/oz480/mcoughli/AppleCider/rtf/runs/ae_dim64_meta_img_ft_gp_randtrunc_cls0.5/best_model.pt \
    --summary /fred/oz480/mcoughli/AppleCider/rtf/runs/ae_dim64_meta_img_ft_gp_randtrunc_cls0.5/summary.json \
    --data-dir /fred/oz480/mcoughli/AppleCider/rtf/data_gp \
    --output-dir /fred/oz480/mcoughli/AppleCider/rtf/analysis/early_zoobot_ft_cls \
    --detection-counts 3 5 7 10 15 20 30 50 100 \
    --image-backbone zoobot \
    --num-classes 5

echo "Done: $(date)"
