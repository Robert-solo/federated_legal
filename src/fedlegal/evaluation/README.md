# Evaluation Module

Metrics for legal, federated, LLM, privacy, and transfer evaluation.

## Architecture

This module enforces the metric surface required by `paper.md` and `AGENTS.md`.

Implemented metrics:

- legal accuracy
- citation accuracy
- citation consistency
- legal consistency
- reasoning coherence
- hallucination rate
- communication cost
- client drift
- cross-jurisdiction generalization
- privacy leakage risk
- aggregation stability
- micro-F1 for multi-label classification
- exact-match / token-F1 for extractive QA

## Usage

```python
from fedlegal.config import load_experiment_config
from fedlegal.evaluation import EvaluationRunner

config = load_experiment_config("configs/experiments/evaluation_default.yaml")
report = EvaluationRunner(config).run(predictions_path="predictions.jsonl")
```

CLI:

```bash
python scripts/run_evaluation.py configs/experiments/evaluation_default.yaml --predictions predictions.jsonl
```

Artifacts are written to `outputs/evaluation/<run_name>/`.

## Published-label diagnostics

`legalbench_diagnostic.py` supplies task-specific prompts, including candidate
citations, and computes accuracy, macro F1, balanced accuracy, multiclass Brier
score and ten-bin top-confidence ECE. It uses benchmark labels; expert authority
conflict and blinded reasoning quality use their separate annotation pipelines.

```bash
python scripts/run_legalbench_diagnostics.py --model /path/to/cached/model --condition qwen_base --output outputs/revision_v2/legalbench/qwen_base
python scripts/analyze_revision_legalbench.py --input outputs/revision_v2/legalbench --output outputs/revision_v2/analysis/legalbench
```

The classifier runner evaluates a fixed sample after checkpoints are frozen.
See `experiments/revision_v2/README.md` for adapter-seed and pretrained comparisons.
