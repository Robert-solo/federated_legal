#!/usr/bin/env bash
#SBATCH --job-name=ipm-reasoning-confirm
#SBATCH --partition=gre
#SBATCH --exclude=gpu12,gpu17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
export HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/src:$PWD"
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
MODEL_PATH=$(find "$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" -name config.json -printf '%h\n' | head -1)
TASK=${SLURM_ARRAY_TASK_ID:-0}
EXTRA=()
CONDITION=base
if [[ "$TASK" -gt 0 ]]; then
  GROUP=fedavg
  SPEC=fedavg_10
  if [[ "$TASK" -gt 5 ]]; then GROUP=flen; SPEC=flen_guarded_8_2; fi
  SEED=$((41+(TASK-1)%5))
  CONDITION="${GROUP}_seed${SEED}"
  EXTRA+=(--adapter "outputs/revision_v2/casehold/${SPEC}_seed${SEED}/shared_adapter")
fi
OUT="outputs/revision_v2/reasoning_confirmation/runs/${CONDITION}"
"$PY" -u scripts/run_multi_agent_blind_eval.py \
  --dataset outputs/revision_v2/reasoning_confirmation/cases.jsonl \
  --model "$MODEL_PATH" --output "$OUT" --limit 200 --seed 41 --single-passes 4 --require-cuda "${EXTRA[@]}"
