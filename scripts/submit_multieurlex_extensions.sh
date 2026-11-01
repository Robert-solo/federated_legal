#!/usr/bin/env bash
set -euo pipefail

cd /remote_dir/home/junluo/IPM

for METHOD in local_only unadapted proxy_flen; do
  for HOLDOUT in en-de en-fr en-es en-pl; do
    JOB_ID=$(sbatch --parsable \
      --export=ALL,METHOD="${METHOD}",HOLDOUT="${HOLDOUT}",SEED=41,RUNS=3,ROUNDS=3,LIMIT=500 \
      --job-name="ipm-multi-${METHOD}-${HOLDOUT}" \
      scripts/slurm_multieurlex_federated.sh)
    echo "Submitted ${METHOD}/${HOLDOUT} as ${JOB_ID}"
  done
done
