# Experiments

Every experiment must include:

- YAML configuration under `configs/experiments`
- a matching experiment folder under `experiments`
- logs under `outputs/logs`
- metrics under `outputs/metrics`
- figures under `outputs/figures`
- reproducibility notes in the experiment README

## Initial Experiment Manifests

- `baseline_fedlora`: Flower + FedAvg-style LoRA adapter federation baseline.
- `conflict_aware`: legacy conflict-diagnostic configuration; the current runtime falls back to FedAvg and cannot support a FLEN effectiveness claim.
- `langgraph_reasoning`: federated multi-agent judicial reasoning workflow skeleton.

## Validation

Run:

```bash
python scripts/validate_config.py configs/experiments/baseline_fedlora.yaml
```

This validates the typed configuration only. It does not download datasets, initialize LLMs, or start Flower.

## Reproducibility Requirements

Each future training run should write:

- resolved config snapshot
- random seed and environment metadata
- client partition summary
- communication payload sizes
- per-round metrics
- final legal and federated evaluation report
- generated figures with source data

The reviewed dataset inventory, licenses, experiment roles, and unresolved human-annotation requirements are tracked in `docs/experiments/dataset_coverage.md`.

The implementation-ready method, fair baseline protocol, staged compute plan, statistical tests, success criteria, and claim-release gates are specified in `papers/ipm/METHOD_AND_EXPERIMENT_COMPLETION_PLAN.md`. This plan supersedes any older instruction that treats unrelated dataset tasks as a cross-jurisdiction accuracy matrix.
