#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

export NUM_CLIENTS="${NUM_CLIENTS:-4}"
export NUM_ROUNDS="${NUM_ROUNDS:-3}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1200}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-240}"
export MAX_STEPS="${MAX_STEPS:-10}"
export PARTITION_STRATEGY="${PARTITION_STRATEGY:-jurisdiction}"
export DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"

declare -a MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "mistralai/Mistral-7B-Instruct-v0.2"
)

BASE_PORT="${BASE_PORT:-18080}"

for INDEX in "${!MODELS[@]}"; do
  MODEL_NAME="${MODELS[$INDEX]}"
  export MODEL_NAME
  SAFE_NAME=$(echo "${MODEL_NAME}" | tr '/:.' '____')
  export RUN_NAME="real_compare_${SAFE_NAME}_casehold_${PARTITION_STRATEGY}_${NUM_ROUNDS}r"
  export SERVER_ADDRESS="127.0.0.1:$((BASE_PORT + INDEX))"
  sbatch scripts/slurm_real_flower_peft.sh
done
