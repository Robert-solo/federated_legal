#!/usr/bin/env bash
#SBATCH --job-name=ipm-v2-multieurlex
#SBATCH --partition=gre
#SBATCH --exclude=gpu12,gpu17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
METHODS=(fedavg fedprox lambda0 proxy_flen local_only unadapted)
HOLDOUTS=(en-de en-fr en-es en-pl)
TASK=${SLURM_ARRAY_TASK_ID:-0}
export METHOD=${METHODS[$((TASK / 4))]}
export HOLDOUT=${HOLDOUTS[$((TASK % 4))]}
export SEED=41 RUNS=5 LIMIT=500 ROUNDS=3 EVALUATION_SEED=20261101
export OUTPUT_PREFIX="revision_v2_multieurlex_${METHOD}_${HOLDOUT}"
export CLIP_UPDATES=0 DP_NOISE_MULTIPLIER=0 WRITE_MEMBERSHIP_FEATURES=0
export MODEL_PATH=$(find "$PWD/.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" -name config.json -printf '%h\n' | head -1)
bash scripts/slurm_multieurlex_federated.sh
