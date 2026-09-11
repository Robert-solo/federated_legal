#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

CONDA_ROOT=/persist_data/home/junluo/anaconda3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate ipm-fedlegal

export PYTHONPATH=src
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BASE_URL="${BASE_URL:-http://gpu33:19527/v1}"
API_KEY="${API_KEY:-empty}"
MODEL="${MODEL:-gpt-oss-120b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/api_baselines}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/experiments/evaluation_default.yaml}"

declare -a TASKS=(
  "casehold data/raw/public/CaseHOLD/case_hold/validation.jsonl 200"
  "scotus data/raw/public/LexGLUE/scotus/validation.jsonl 200"
  "cail data/raw/public/CAIL/cail2018/exercise_contest_valid.jsonl 120"
  "cuad_qa data/raw/public/CUAD/cuad_qa/test.jsonl 120"
  "ecthr_a data/raw/public/LexGLUE/ecthr_a/validation.jsonl 100"
  "ecthr_b data/raw/public/LexGLUE/ecthr_b/validation.jsonl 100"
  "eurlex data/raw/public/LexGLUE/eurlex/validation.jsonl 100"
  "ledgar data/raw/public/LexGLUE/ledgar/validation.jsonl 100"
  "unfair_tos data/raw/public/LexGLUE/unfair_tos/validation.jsonl 100"
)

for spec in "${TASKS[@]}"; do
  read -r task dataset_path max_samples <<<"${spec}"
  run_name="gpt_oss_120b_${task}_paper_${max_samples}"
  python scripts/run_api_model_baseline.py \
    --base-url "${BASE_URL}" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    --task "${task}" \
    --dataset-path "${dataset_path}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-name "${run_name}" \
    --max-samples "${max_samples}" \
    --reasoning-effort high \
    --max-tokens 512 \
    --timeout 240

  python scripts/run_evaluation.py \
    "${EVAL_CONFIG}" \
    --predictions "${OUTPUT_ROOT}/${run_name}/predictions.jsonl" \
    --communication-log "${OUTPUT_ROOT}/${run_name}/communication_log.jsonl" || true
done

python scripts/summarize_experiments.py --root outputs --output-dir outputs/evaluation/paper_summary
