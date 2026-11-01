#!/usr/bin/env bash
#SBATCH --job-name=ipm-v2-flen-agents
#SBATCH --partition=gre
#SBATCH --exclude=gpu12,gpu17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail
cd /remote_dir/home/junluo/IPM
export HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/src:$PWD"
PY=/persist_data/home/junluo/anaconda3/envs/ipm-fedlegal/bin/python
MODEL_PATH=$(find "$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" -name config.json -printf '%h\n' | head -1)
for PASSES in 1 4; do
  OUT="outputs/revision_v2/human_review/flen_${PASSES}pass"
  "$PY" -u scripts/run_multi_agent_blind_eval.py --dataset data/raw/public/CaseHOLD/case_hold/test.jsonl \
    --model "$MODEL_PATH" --adapter outputs/revision_v2/casehold/flen_guarded_8_2_seed41/shared_adapter \
    --output "$OUT" --limit 60 --seed 41 --single-passes "$PASSES" --require-cuda
  "$PY" scripts/analyze_multi_agent_blind_eval.py --predictions "$OUT/predictions.jsonl" --output "$OUT/automatic_analysis.json"
  cp "$OUT/review_sheet.csv" "$OUT/reviewer_1_sheet.csv"
  cp "$OUT/review_sheet.csv" "$OUT/reviewer_2_sheet.csv"
done
