#!/usr/bin/env bash
#SBATCH --job-name=ipm-v2-analysis
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
"$PY" scripts/analyze_revision_v2.py --kind casehold --input outputs/revision_v2/casehold --output outputs/revision_v2/analysis/casehold
"$PY" scripts/analyze_revision_v2.py --kind privacy --input outputs/experiments --output outputs/revision_v2/analysis/privacy
