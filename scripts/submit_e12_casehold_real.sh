#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

SEED_VALUE=${SEED:-42}
RUN_NAME_VALUE=${RUN_NAME:-E12_casehold_real_seed${SEED_VALUE}_10r}
RUN_DIR="outputs/real_flower_peft/${RUN_NAME_VALUE}"

if [[ -e "${RUN_DIR}/server_rounds.jsonl" || -e "${RUN_DIR}/real_experiment_report.json" ]]; then
  echo "Refusing to append to existing run artifacts: ${RUN_DIR}" >&2
  echo "Set a new RUN_NAME or archive the existing run before submission." >&2
  exit 2
fi

JOB_ID=$(sbatch --parsable \
  --job-name=ipm-e12-casehold \
  --export="ALL,RUN_NAME=${RUN_NAME_VALUE},MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct,NUM_CLIENTS=4,NUM_ROUNDS=10,MAX_TRAIN_SAMPLES=1200,MAX_EVAL_SAMPLES=240,MAX_STEPS=10,PARTITION_STRATEGY=jurisdiction,DIRICHLET_ALPHA=0.3,SEED=${SEED_VALUE}" \
  scripts/slurm_real_flower_peft.sh)

echo "Submitted E12 run ${RUN_NAME_VALUE} as Slurm job ${JOB_ID}"
echo "Expected acceptance artifact: ${RUN_DIR}/acceptance.json"
