#!/usr/bin/env bash
#SBATCH --job-name=ipm-agent-blind
#SBATCH --partition=gre
#SBATCH --exclude=gpu12,gpu17
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
cd /remote_dir/home/junluo/IPM
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
MODEL_PATH=${MODEL_PATH:-/remote_dir/home/junluo/IPM/.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
LIMIT=${LIMIT:-60}
SEED=${SEED:-41}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/human_review/casehold_multi_agent_seed${SEED}}
mkdir -p outputs/slurm

"$PY" -u scripts/run_multi_agent_blind_eval.py \
  --dataset data/raw/public/CaseHOLD/case_hold/validation.jsonl \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_DIR" \
  --limit "$LIMIT" --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" --require-cuda
