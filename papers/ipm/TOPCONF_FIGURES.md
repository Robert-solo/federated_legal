# Top-Conference Experiment Figure Templates

These figures are structured as a quantitative evidence chain for the revised experimental section. The generated numbers are synthetic layout data until replaced with measured runs.

| Figure | Scientific question | Panels | Required real inputs |
|---|---|---|---|
| `fig_exp01_benchmark` | Does the framework improve primary legal task performance against controlled baselines? | Dataset-wise accuracy, aggregated mean/CI, gain versus FedAvg | Five-seed legal accuracy for all compared methods and datasets |
| `fig_exp02_reliability` | Are gains legally reliable rather than accuracy-only? | Reliability heatmap, citation F1 versus hallucination trade-off | Citation F1, reasoning coherence, hallucination rate |
| `fig_exp03_dynamics` | Does federated optimization converge efficiently? | Accuracy, client drift, cumulative communication across rounds | Round-level multi-seed metrics and transmitted bytes |
| `fig_exp04_conflict` | Does the method suppress operational legal conflict? | Weighted conflict curve, round-100 component profile | Four conflict scores and weighted total per round |
| `fig_exp05_transfer` | Does the framework transfer across unseen jurisdictions? | Transfer matrix, held-out-jurisdiction comparison | Train/test-jurisdiction test results |
| `fig_exp06_ablation` | Which modules account for performance and reliability? | Accuracy, citation, hallucination deltas versus full model | Paired ablation results on identical seeds/splits |
| `fig_exp07_privacy` | What utility and cost accompanies privacy protection? | Utility, membership-inference AUC, communication overhead | DP sweeps and secure-aggregation measurements |
| `fig_exp08_robustness` | Is the method robust to scale and non-IID severity? | Sensitivity heatmap, scale curves, communication growth | Client-count and Dirichlet-alpha grid |

## Compared Models

The primary benchmark template includes:

`Centralized`, `Local-only`, `FedAvg`, `FedProx`, `SCAFFOLD`, `FedNova`, `FedLoRA`, `Conflict-aware`, and `Full framework`.

Training dynamics uses only federated approaches because centralized and local-only methods do not have server communication rounds. Transfer and robustness plots retain the controlled or deployment-relevant subset to keep panels readable.

## Generate Templates

```bash
python scripts/generate_topconf_figure_templates.py
```

The command writes source CSV templates to:

```text
tables/experiment_source_data_topconf/
```

and exports editable `PDF`/`SVG` plus high-resolution `PNG` previews to:

```text
figures/experiments_topconf/
```

## Insert Real Results

Replace the CSV contents while keeping the column names, then render with:

```bash
python scripts/generate_topconf_figure_templates.py --use-existing-data --final-data
```

Without `--final-data`, every figure remains visibly marked as synthetic template data.
