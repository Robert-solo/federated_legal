#!/usr/bin/env bash
#SBATCH --job-name=ipm-e12-predict
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /remote_dir/home/junluo/IPM
mkdir -p outputs/slurm

source /persist_data/home/junluo/anaconda3/etc/profile.d/conda.sh
conda activate ipm-fedlegal
export PYTHONNOUSERSITE=1
export PYTHONPATH=src
export HF_HOME=/remote_dir/home/junluo/IPM/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

python scripts/run_real_flower_peft.py \
  --mode export-predictions \
  --run-name E12_casehold_real_10r_20260909_seed42 \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --num-clients 4 \
  --num-rounds 10 \
  --max-train-samples 1200 \
  --max-eval-samples 240 \
  --max-steps 10 \
  --max-length 512 \
  --partition-strategy jurisdiction \
  --seed 42 \
  --method fedavg \
  --client-id 0 \
  --checkpoint-round 10
