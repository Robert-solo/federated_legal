#!/usr/bin/env bash
#SBATCH --job-name=ipm-legal-data
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /remote_dir/home/junluo/IPM
mkdir -p outputs/slurm data/raw/public_extended

CONDA_ROOT=/persist_data/home/junluo/anaconda3
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate ipm-fedlegal

export PYTHONNOUSERSITE=1
export HF_HOME=/remote_dir/home/junluo/IPM/.hf_cache
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python scripts/download_public_legal_datasets.py \
  --output-dir data/raw/public_extended \
  --extended-only \
  --skip-existing
