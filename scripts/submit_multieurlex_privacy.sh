#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

for SPEC in no_privacy:0.0 gaussian_0_5:0.5 gaussian_1_0:1.0; do
  NAME=${SPEC%%:*}
  NOISE=${SPEC##*:}
  PREFIX="multieurlex_privacy_strict_${NAME}_en-pl"
  JOB_ID=$(sbatch --parsable \
    --export=ALL,METHOD=fedavg,HOLDOUT=en-pl,SEED=41,RUNS=3,ROUNDS=3,LIMIT=500,DP_CLIP_NORM=1.0,DP_NOISE_MULTIPLIER="${NOISE}",DP_DELTA=0.00001,WRITE_MEMBERSHIP_FEATURES=1,OUTPUT_PREFIX="${PREFIX}" \
    --job-name="ipm-privacy-${NAME}" \
    scripts/slurm_multieurlex_federated.sh)
  echo "Submitted ${NAME} as ${JOB_ID}"
done
