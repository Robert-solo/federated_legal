#!/usr/bin/env bash
#SBATCH --job-name=ipm-revision-v2
#SBATCH --partition=gre
#SBATCH --exclude=gpu12,gpu17
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
export HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD"
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
SPECS=(fedavg_10 fedprox_10 ditto_8_2 unguarded_8_2 guarded_8_2 unguarded_10_2 guarded_10_2 flen_guarded_8_2)
TASK=${SLURM_ARRAY_TASK_ID:-0}
SPEC=${SPEC:-${SPECS[$((TASK / 5))]}}
SEED=${SEED:-$((41 + TASK % 5))}
MODEL_PATH=${MODEL_PATH:-$(find "$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" -name config.json -printf '%h\n' | head -1)}
"$PY" -u scripts/run_casehold_revision.py --spec "$SPEC" --seed "$SEED" --model "$MODEL_PATH" \
  --rounds "${ROUNDS:-20}" --test-limit "${TEST_LIMIT:-1000}" --train-limit "${TRAIN_LIMIT:-1200}" \
  --validation-limit "${VALIDATION_LIMIT:-80}" --output-root "${OUTPUT_ROOT:-outputs/revision_v2/casehold}"
