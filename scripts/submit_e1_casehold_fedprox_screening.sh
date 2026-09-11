#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

SEEDS_VALUE=${SEEDS:-"41 42 43"}
RUN_PREFIX=${RUN_PREFIX:-E1_casehold_fedprox_20r}
FEDPROX_MU_VALUE=${FEDPROX_MU:-0.01}

for SEED_VALUE in ${SEEDS_VALUE}; do
  RUN_NAME_VALUE="${RUN_PREFIX}_seed${SEED_VALUE}"
  RUN_DIR="outputs/real_flower_peft/${RUN_NAME_VALUE}"
  if [[ -e "${RUN_DIR}/server_rounds.jsonl" || -e "${RUN_DIR}/real_experiment_report.json" ]]; then
    echo "Refusing to append to existing run artifacts: ${RUN_DIR}" >&2
    exit 2
  fi

  JOB_ID=$(sbatch --parsable \
    --job-name="ipm-e1-fedprox-s${SEED_VALUE}" \
    --export="ALL,RUN_NAME=${RUN_NAME_VALUE},MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct,NUM_CLIENTS=4,NUM_ROUNDS=20,MAX_TRAIN_SAMPLES=1200,MAX_EVAL_SAMPLES=240,MAX_STEPS=10,PARTITION_STRATEGY=jurisdiction,DIRICHLET_ALPHA=0.3,SEED=${SEED_VALUE},METHOD=fedprox,FEDPROX_MU=${FEDPROX_MU_VALUE}" \
    scripts/slurm_real_flower_peft.sh)
  echo "Submitted ${RUN_NAME_VALUE} as Slurm job ${JOB_ID}"
done
