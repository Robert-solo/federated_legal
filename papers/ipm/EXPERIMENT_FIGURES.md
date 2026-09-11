# Experiment Figure Plan

The experiment section should use figures to answer four reviewer-facing questions: whether the framework improves legal performance, why it improves, whether it generalizes across jurisdictions, and what privacy/system cost it introduces.

## Required Figure Templates

1. `exp1_main_performance`
   - Claim: the full framework should outperform centralized, local-only, standard federated, and PEFT baselines across legal datasets.
   - Models: Centralized, Local-only, FedAvg, FedProx, SCAFFOLD, FedNova, FedLoRA, Conflict-aware, Full framework.
   - Metrics: legal accuracy with standard error.

2. `exp2_legal_reliability`
   - Claim: gains should include legal reliability, not only task accuracy.
   - Metrics: citation consistency, reasoning coherence, hallucination rate.

3. `exp3_training_dynamics`
   - Claim: the method should converge stably and reduce client drift.
   - Metrics: per-round legal accuracy and client drift.

4. `exp4_conflict_diagnostics`
   - Claim: conflict-aware aggregation should lower legal conflict signals.
   - Metrics: citation conflict, reasoning conflict, verdict conflict, rule-alignment distance, total conflict.

5. `exp5_cross_jurisdiction`
   - Claim: jurisdiction-aware modelling should improve held-out jurisdiction transfer.
   - Metrics: train-test jurisdiction accuracy matrix and held-out average.

6. `exp6_ablation`
   - Claim: each major module contributes to performance and reliability.
   - Variants: no jurisdiction embedding, no citation penalty, no reasoning penalty, no verdict penalty, no rule alignment, no citation verifier, no debate agents, no privacy auditor.

7. `exp7_privacy_utility`
   - Claim: privacy controls introduce a measurable but manageable utility trade-off.
   - Metrics: legal accuracy and leakage risk under different DP noise multipliers and secure aggregation settings.

8. `exp8_scaling_sensitivity`
   - Claim: the method should remain robust as client count and non-IID severity change.
   - Metrics: accuracy across number of clients and Dirichlet alpha.

## How to Replace Data

Each figure has a CSV under:

```text
tables/experiment_source_data/
```

Replace the values while preserving column names, then run:

```bash
python scripts/generate_experiment_figure_templates.py --use-existing-data
```

The regenerated figures are written to:

```text
figures/experiments/
```

By default, every rendered figure is marked `SYNTHETIC PLACEHOLDER DATA - REPLACE BEFORE SUBMISSION`. After all CSVs have been replaced with verified experimental results, remove this marker only with:

```bash
python scripts/generate_experiment_figure_templates.py --use-existing-data --final-data
```
