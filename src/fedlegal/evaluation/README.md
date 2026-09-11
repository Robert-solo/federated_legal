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
