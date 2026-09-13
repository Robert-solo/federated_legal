#!/usr/bin/env bash
#SBATCH --job-name=ipm-multieurlex
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd /remote_dir/home/junluo/IPM
source /persist_data/home/junluo/anaconda3/etc/profile.d/conda.sh
conda activate ipm-fedlegal
export HF_HOME=/remote_dir/home/junluo/IPM/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
mkdir -p outputs/slurm

HOLDOUT=${HOLDOUT:-en-pl}
METHOD=${METHOD:-fedavg}
SEED=${SEED:-41}
RUNS=${RUNS:-3}
LIMIT=${LIMIT:-500}
ROUNDS=${ROUNDS:-3}
for RUN_SEED in $(seq "${SEED}" $((SEED + RUNS - 1))); do
  python scripts/run_multieurlex_federated.py \
    --raw-root data/raw/public_extended/MultiEURLEX \
    --output "outputs/experiments/multieurlex_${METHOD}_${HOLDOUT}_seed${RUN_SEED}" \
    --holdout "${HOLDOUT}" --method "${METHOD}" --seed "${RUN_SEED}" \
    --rounds "${ROUNDS}" --limit-per-client "${LIMIT}" --eval-limit "${LIMIT}" \
    --model Qwen/Qwen2.5-0.5B-Instruct --max-length 256 --batch-size 4
done
