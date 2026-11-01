#!/usr/bin/env bash
set -euo pipefail
cd /remote_dir/home/junluo/IPM
for SPEC in no_clip:0:0 clipped_only:1:0 gaussian_0_25:1:0.25 gaussian_0_5:1:0.5 gaussian_1_0:1:1.0; do
  IFS=: read -r NAME CLIP NOISE <<< "$SPEC"
  sbatch --parsable --job-name="ipm-v2-${NAME}" \
    --export="ALL,METHOD=fedavg,HOLDOUT=en-pl,SEED=41,RUNS=3,ROUNDS=3,LIMIT=500,CLIP_UPDATES=${CLIP},DP_CLIP_NORM=1.0,DP_NOISE_MULTIPLIER=${NOISE},WRITE_MEMBERSHIP_FEATURES=1,OUTPUT_PREFIX=revision_v2_privacy_${NAME}_en-pl" \
    scripts/slurm_multieurlex_federated.sh
done
