#!/usr/bin/env bash
#SBATCH --job-name=ipm-reasoning-confirm-analysis
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
export PYTHONPATH="$PWD/src:$PWD"
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
"$PY" scripts/analyze_reasoning_confirmation.py
"$PY" scripts/export_revision_evidence.py --output outputs/revision_v2/revision_evidence_confirmation.tar.gz
