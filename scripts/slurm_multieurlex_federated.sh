#!/usr/bin/env bash
#SBATCH --job-name=ipm-multieurlex
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
DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
DP_NOISE_MULTIPLIER=${DP_NOISE_MULTIPLIER:-0.0}
DP_DELTA=${DP_DELTA:-0.00001}
WRITE_MEMBERSHIP_FEATURES=${WRITE_MEMBERSHIP_FEATURES:-0}
CLIP_UPDATES=${CLIP_UPDATES:-0}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-multieurlex_${METHOD}_${HOLDOUT}}
for RUN_SEED in $(seq "${SEED}" $((SEED + RUNS - 1))); do
  MODEL_PATH=${MODEL_PATH:-/remote_dir/home/junluo/IPM/.hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775}
  RUN_OUTPUT="outputs/experiments/${OUTPUT_PREFIX}_seed${RUN_SEED}"
  EXTRA_ARGS=()
  if [[ "${CLIP_UPDATES}" == "1" ]]; then
    EXTRA_ARGS+=(--clip-updates)
  fi
  if [[ "${WRITE_MEMBERSHIP_FEATURES}" == "1" ]]; then
    EXTRA_ARGS+=(--write-membership-features)
  fi
  "$PY" -u scripts/run_multieurlex_federated.py \
    --raw-root data/raw/public_extended/MultiEURLEX \
    --output "${RUN_OUTPUT}" \
    --holdout "${HOLDOUT}" --method "${METHOD}" --seed "${RUN_SEED}" \
    --rounds "${ROUNDS}" --limit-per-client "${LIMIT}" --eval-limit "${LIMIT}" \
    --model "${MODEL_PATH}" --max-length 256 --batch-size 4 --require-cuda \
    --dp-clip-norm "${DP_CLIP_NORM}" \
    --dp-noise-multiplier "${DP_NOISE_MULTIPLIER}" --dp-delta "${DP_DELTA}" \
    --test-final-only \
    "${EXTRA_ARGS[@]}"
  if [[ "${WRITE_MEMBERSHIP_FEATURES}" == "1" ]]; then
    "$PY" -u scripts/run_membership_inference_attack.py \
      --features "${RUN_OUTPUT}/membership_features.jsonl" \
      --output "${RUN_OUTPUT}/privacy_attack" --seed "${RUN_SEED}"
  fi
done
