#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

SEEDS_VALUE=${SEEDS:-"41 42 43"}
NUM_ROUNDS_VALUE=${NUM_ROUNDS:-20}
MAX_TRAIN_SAMPLES_VALUE=${MAX_TRAIN_SAMPLES:-1200}
MAX_EVAL_SAMPLES_VALUE=${MAX_EVAL_SAMPLES:-240}
MAX_PROBE_SAMPLES_VALUE=${MAX_PROBE_SAMPLES:-24}
RUN_TAG=${RUN_TAG:-20260901_v1}

for METHOD_VALUE in personalization_guarded flen_guarded; do
  for SEED_VALUE in ${SEEDS_VALUE}; do
    if [[ "${METHOD_VALUE}" == "flen_guarded" ]]; then
      REQUIRE_CONFLICT_ACTIVATION_VALUE=1
    else
      REQUIRE_CONFLICT_ACTIVATION_VALUE=0
    fi
    RUN_NAME_VALUE="E1_casehold_${METHOD_VALUE}_${NUM_ROUNDS_VALUE}r_${RUN_TAG}_seed${SEED_VALUE}"
    RUN_DIR="outputs/real_flower_peft/${RUN_NAME_VALUE}"
    if [[ -e "${RUN_DIR}/server_rounds.jsonl" || -e "${RUN_DIR}/real_experiment_report.json" ]]; then
      echo "Refusing to append to existing run artifacts: ${RUN_DIR}" >&2
      exit 2
    fi
    JOB_ID=$(sbatch --parsable \
      --job-name="ipm-e1-${METHOD_VALUE}-s${SEED_VALUE}" \
      --export="ALL,RUN_NAME=${RUN_NAME_VALUE},MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct,NUM_CLIENTS=4,NUM_ROUNDS=${NUM_ROUNDS_VALUE},MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES_VALUE},MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES_VALUE},MAX_PROBE_SAMPLES=${MAX_PROBE_SAMPLES_VALUE},MAX_STEPS=10,LOCAL_PERSONALIZATION_STEPS=2,PARTITION_STRATEGY=jurisdiction,DIRICHLET_ALPHA=0.3,SEED=${SEED_VALUE},METHOD=${METHOD_VALUE},CONFLICT_LAMBDA=0.2,DIVERSITY_FLOOR=0.1,WEIGHT_SMOOTHING=0.5,SAMPLE_COUNT_CAP=1000,REQUIRE_EXPERT_CALIBRATION=0,REQUIRE_CONFLICT_ACTIVATION=${REQUIRE_CONFLICT_ACTIVATION_VALUE}" \
      scripts/slurm_real_flower_peft.sh)
    echo "Submitted ${RUN_NAME_VALUE} as Slurm job ${JOB_ID}"
  done
done
