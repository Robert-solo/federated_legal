#!/usr/bin/env bash
#SBATCH --job-name=ipm-fedavg-5seed
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /remote_dir/home/junluo/IPM
mkdir -p outputs/slurm

SEEDS_VALUE=${SEEDS:-"41 42 43 44 45"}
RUN_PREFIX=${RUN_PREFIX:-paper_casehold_fedavg_jurisdiction}
PYTHON_BIN=${PYTHON_BIN:-/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python}

for SEED_VALUE in ${SEEDS_VALUE}; do
  RUN_NAME="${RUN_PREFIX}_seed${SEED_VALUE}" \
  SEED="${SEED_VALUE}" \
  MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" \
  NUM_CLIENTS=4 \
  NUM_ROUNDS=3 \
  MAX_TRAIN_SAMPLES=1200 \
  MAX_EVAL_SAMPLES=240 \
  MAX_STEPS=10 \
  PARTITION_STRATEGY=jurisdiction \
  DIRICHLET_ALPHA=0.3 \
  bash scripts/slurm_real_flower_peft.sh
done

"${PYTHON_BIN}" scripts/summarize_experiments.py \
  --root outputs \
  --output-dir outputs/evaluation/paper_casehold_fedavg_5seed
