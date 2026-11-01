#!/usr/bin/env bash
#SBATCH --job-name=ipm-v2-legalbench
#SBATCH --partition=gre
#SBATCH --exclude=gpu12,gpu17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
export HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/src:$PWD"
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
GROUP=${GROUP:-base}
INDEX=${SLURM_ARRAY_TASK_ID:-0}
EXTRA=()
MODEL_NAME=Qwen--Qwen2.5-0.5B-Instruct
if [[ "$GROUP" == base ]]; then
  CONDITION=qwen_base
  if [[ "$INDEX" == 1 ]]; then CONDITION=mistral_7b_base; MODEL_NAME=mistralai--Mistral-7B-Instruct-v0.2; fi
else
  SEED=$((41+INDEX))
  CONDITION="qwen_${GROUP}_seed${SEED}"
  SPEC=fedavg_10
  if [[ "$GROUP" == flen ]]; then SPEC=flen_guarded_8_2; fi
  EXTRA+=(--adapter "outputs/revision_v2/casehold/${SPEC}_seed${SEED}/shared_adapter")
  EXTRA+=(--training-seed "$SEED")
fi
MODEL_PATH=$(find "$HF_HOME/hub/models--${MODEL_NAME}/snapshots" -name config.json -printf '%h\n' | head -1)
"$PY" -u scripts/run_legalbench_diagnostics.py --model "$MODEL_PATH" --condition "$CONDITION" \
  --output "${OUTPUT_ROOT:-outputs/revision_v2/legalbench}/${CONDITION}" --limit "${LIMIT:-240}" "${EXTRA[@]}"
