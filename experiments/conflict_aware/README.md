# Conflict-Aware Aggregation Experiment

## Goal

Evaluate federated LoRA aggregation with legal conflict penalties across citation, reasoning, verdict, and rule-alignment dimensions.

## Config

`configs/experiments/conflict_aware.yaml`

The executable CaseHOLD proxy-screening contract is
`experiments/conflict_aware/casehold_flen_proxy_screening.yaml`.
The component-isolation contract is
`experiments/conflict_aware/casehold_flen_ablation.yaml`.
The guarded local-residual contract is
`experiments/conflict_aware/casehold_flen_guarded.yaml`.

## Planned Outputs

- logs: `outputs/logs/conflict_aware`
- metrics: `outputs/metrics/conflict_aware`
- figures: `outputs/figures/conflict_aware`

## Reproducibility

```bash
python scripts/validate_config.py configs/experiments/conflict_aware.yaml
bash scripts/submit_e1_casehold_flen_screening.sh
bash scripts/submit_e1_casehold_flen_ablation.sh
bash scripts/submit_e1_casehold_flen_guarded.sh
```

The Flower runtime implements target-conditioned shared LoRA aggregation, a
lambda-zero target-cohort control, calibrated proxy metadata, disjoint public probes,
gold-error-gated pairwise verdict disagreement, diversity floors, temporal weight
smoothing, local FLEN residuals, checkpoints, communication logs, and run summaries.

The bundled CaseHOLD calibration is an `automatic_proxy_pilot`. It supports runtime
and mechanism screening only and cannot support formal legal-conflict or authentic
cross-jurisdiction claims. Formal studies replace it with expert-adjudicated metadata.
