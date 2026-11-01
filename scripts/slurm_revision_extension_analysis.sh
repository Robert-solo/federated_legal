#!/usr/bin/env bash
#SBATCH --job-name=ipm-v2-extension-analysis
#SBATCH --partition=gre
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
export PYTHONPATH="$PWD/src:$PWD"
"$PY" scripts/analyze_revision_v2.py --kind casehold --input outputs/revision_v2/casehold \
  --fedadam-input outputs/revision_v2/casehold_fedadam --output outputs/revision_v2/analysis/casehold_extended
"$PY" scripts/analyze_revision_multieurlex.py --input outputs/experiments --output outputs/revision_v2/analysis/multieurlex
"$PY" scripts/analyze_revision_legalbench.py --input outputs/revision_v2/legalbench --output outputs/revision_v2/analysis/legalbench
"$PY" scripts/analyze_revision_agents.py --input outputs/revision_v2/human_review --output outputs/revision_v2/analysis/agents
"$PY" scripts/publish_revision_results.py --analysis-root outputs/revision_v2/analysis
"$PY" scripts/export_revision_evidence.py --output outputs/revision_v2/revision_evidence.tar.gz
