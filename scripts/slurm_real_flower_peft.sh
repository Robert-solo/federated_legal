#!/usr/bin/env bash
#SBATCH --job-name=ipm-real-peft
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
set -x

cd /remote_dir/home/junluo/IPM
mkdir -p outputs/slurm

CONDA_ROOT=/persist_data/home/junluo/anaconda3
ENV_NAME=ipm-fedlegal

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"

ENV_READY=.slurm_env/ipm-fedlegal.ready
mkdir -p "$(dirname "${ENV_READY}")"
if [[ ! -f "${ENV_READY}" ]]; then
  python -m pip install --upgrade pip
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.1"
  python -m pip install \
    "transformers==4.46.3" \
    "peft==0.13.2" \
    "accelerate==1.1.1" \
    "datasets==3.1.0" \
    "flwr==1.13.1" \
    "pydantic>=2.6" \
    "PyYAML>=6.0" \
    "numpy<2" \
    "scikit-learn"
  touch "${ENV_READY}"
fi
INSTALL_LOCK=.slurm_env/ipm-fedlegal-install.lock
(
  flock -x 9
  python -m pip install -e . --no-deps
) 9>"${INSTALL_LOCK}"

export HF_HOME=/remote_dir/home/junluo/IPM/.hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1

RUN_NAME=${RUN_NAME:-real_qwen25_casehold_flower}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}
NUM_CLIENTS=${NUM_CLIENTS:-4}
NUM_ROUNDS=${NUM_ROUNDS:-3}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-1200}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-240}
MAX_STEPS=${MAX_STEPS:-10}
PARTITION_STRATEGY=${PARTITION_STRATEGY:-jurisdiction}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.3}
SEED=${SEED:-42}
METHOD=${METHOD:-fedavg}
FEDPROX_MU=${FEDPROX_MU:-0.01}
MAX_PROBE_SAMPLES=${MAX_PROBE_SAMPLES:-24}
CONFLICT_LAMBDA=${CONFLICT_LAMBDA:-0.2}
DIVERSITY_FLOOR=${DIVERSITY_FLOOR:-0.1}
WEIGHT_SMOOTHING=${WEIGHT_SMOOTHING:-0.5}
SAMPLE_COUNT_CAP=${SAMPLE_COUNT_CAP:-1000}
LOCAL_PERSONALIZATION_STEPS=${LOCAL_PERSONALIZATION_STEPS:-0}
REQUIRE_EXPERT_CALIBRATION=${REQUIRE_EXPERT_CALIBRATION:-0}
REQUIRE_CONFLICT_ACTIVATION=${REQUIRE_CONFLICT_ACTIVATION:-0}
SERVER_ADDRESS=${SERVER_ADDRESS:-127.0.0.1:8080}
RUN_DIR="outputs/real_flower_peft/${RUN_NAME}"
EXPERT_CALIBRATION_ARGS=()
CONFLICT_ACTIVATION_ARGS=()
if [[ "${REQUIRE_EXPERT_CALIBRATION}" == "1" ]]; then
  EXPERT_CALIBRATION_ARGS+=(--require-expert-calibration)
fi
if [[ "${REQUIRE_CONFLICT_ACTIVATION}" == "1" ]]; then
  CONFLICT_ACTIVATION_ARGS+=(--require-conflict-activation)
fi

if [[ ! -f "${RUN_DIR}/model_cache_manifest.json" ]]; then
  python scripts/run_real_flower_peft.py \
    --mode prepare-model \
    --run-name "${RUN_NAME}" \
    --model-name "${MODEL_NAME}" \
    --num-clients "${NUM_CLIENTS}" \
    --num-rounds "${NUM_ROUNDS}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
    --max-steps "${MAX_STEPS}" \
    --max-probe-samples "${MAX_PROBE_SAMPLES}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --fedprox-mu "${FEDPROX_MU}"
fi

if [[ ! -f "${RUN_DIR}/manifest.json" ]]; then
  python scripts/run_real_flower_peft.py \
    --mode prepare-data \
    --run-name "${RUN_NAME}" \
    --model-name "${MODEL_NAME}" \
    --num-clients "${NUM_CLIENTS}" \
    --num-rounds "${NUM_ROUNDS}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
    --partition-strategy "${PARTITION_STRATEGY}" \
    --dirichlet-alpha "${DIRICHLET_ALPHA}" \
    --max-steps "${MAX_STEPS}" \
    --max-probe-samples "${MAX_PROBE_SAMPLES}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --conflict-lambda "${CONFLICT_LAMBDA}" \
    --diversity-floor "${DIVERSITY_FLOOR}" \
    --weight-smoothing "${WEIGHT_SMOOTHING}" \
    --sample-count-cap "${SAMPLE_COUNT_CAP}" \
    --local-personalization-steps "${LOCAL_PERSONALIZATION_STEPS}" \
    "${EXPERT_CALIBRATION_ARGS[@]}"
fi

export HF_HUB_OFFLINE=1

python scripts/run_real_flower_peft.py \
  --mode server \
  --run-name "${RUN_NAME}" \
  --model-name "${MODEL_NAME}" \
  --num-clients "${NUM_CLIENTS}" \
  --num-rounds "${NUM_ROUNDS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --partition-strategy "${PARTITION_STRATEGY}" \
  --dirichlet-alpha "${DIRICHLET_ALPHA}" \
  --max-steps "${MAX_STEPS}" \
  --max-probe-samples "${MAX_PROBE_SAMPLES}" \
  --seed "${SEED}" \
  --method "${METHOD}" \
  --conflict-lambda "${CONFLICT_LAMBDA}" \
  --diversity-floor "${DIVERSITY_FLOOR}" \
  --weight-smoothing "${WEIGHT_SMOOTHING}" \
  --sample-count-cap "${SAMPLE_COUNT_CAP}" \
  --local-personalization-steps "${LOCAL_PERSONALIZATION_STEPS}" \
  "${EXPERT_CALIBRATION_ARGS[@]}" \
  --server-address "${SERVER_ADDRESS}" &
SERVER_PID=$!

sleep 20

for CLIENT_ID in $(seq 0 $((NUM_CLIENTS - 1))); do
  CUDA_VISIBLE_DEVICES="${CLIENT_ID}" python scripts/run_real_flower_peft.py \
    --mode client \
    --run-name "${RUN_NAME}" \
    --model-name "${MODEL_NAME}" \
    --num-clients "${NUM_CLIENTS}" \
    --client-id "${CLIENT_ID}" \
    --num-rounds "${NUM_ROUNDS}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --max-eval-samples "${MAX_EVAL_SAMPLES}" \
    --partition-strategy "${PARTITION_STRATEGY}" \
    --dirichlet-alpha "${DIRICHLET_ALPHA}" \
    --max-steps "${MAX_STEPS}" \
    --max-probe-samples "${MAX_PROBE_SAMPLES}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    --fedprox-mu "${FEDPROX_MU}" \
    --conflict-lambda "${CONFLICT_LAMBDA}" \
    --diversity-floor "${DIVERSITY_FLOOR}" \
    --weight-smoothing "${WEIGHT_SMOOTHING}" \
    --sample-count-cap "${SAMPLE_COUNT_CAP}" \
    --local-personalization-steps "${LOCAL_PERSONALIZATION_STEPS}" \
    "${EXPERT_CALIBRATION_ARGS[@]}" \
    --server-address "${SERVER_ADDRESS}" &
done

wait "${SERVER_PID}"
wait

python scripts/run_real_flower_peft.py \
  --mode report \
  --run-name "${RUN_NAME}" \
  --model-name "${MODEL_NAME}" \
  --num-clients "${NUM_CLIENTS}" \
  --num-rounds "${NUM_ROUNDS}" \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-eval-samples "${MAX_EVAL_SAMPLES}" \
  --partition-strategy "${PARTITION_STRATEGY}" \
  --dirichlet-alpha "${DIRICHLET_ALPHA}" \
  --max-steps "${MAX_STEPS}" \
  --max-probe-samples "${MAX_PROBE_SAMPLES}" \
  --seed "${SEED}" \
  --method "${METHOD}" \
  --fedprox-mu "${FEDPROX_MU}" \
  --conflict-lambda "${CONFLICT_LAMBDA}" \
  --diversity-floor "${DIVERSITY_FLOOR}" \
  --weight-smoothing "${WEIGHT_SMOOTHING}" \
  --sample-count-cap "${SAMPLE_COUNT_CAP}" \
  --local-personalization-steps "${LOCAL_PERSONALIZATION_STEPS}" \
  "${EXPERT_CALIBRATION_ARGS[@]}"

python scripts/validate_real_flower_run.py \
  "${RUN_DIR}" \
  --expected-rounds "${NUM_ROUNDS}" \
  --expected-clients "${NUM_CLIENTS}" \
  --expected-method "${METHOD}" \
  "${CONFLICT_ACTIVATION_ARGS[@]}" \
  --output "${RUN_DIR}/acceptance.json"
