#!/usr/bin/env bash
#SBATCH --job-name=ipm-reference-weights
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python -u scripts/complete_legalbench_model_cache.py
