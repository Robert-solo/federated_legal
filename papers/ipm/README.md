# IPM LaTeX Paper Package

This folder contains a buildable Information Processing & Management paper structure for:

**Federated Legal Large Language Models for Privacy-Preserving Cross-Jurisdiction Judicial Reasoning**

## Contents

- `main.tex`: IPM manuscript with abstract, introduction, related work, methodology, experiments, and discussion.
- `custom.bib`: bibliography.
- `elsarticle.cls`: Elsevier article class used by the manuscript.
- `figures/`: generated paper figures copied or generated from repository outputs.
- `tables/`: generated LaTeX tables aligned with current experiment artifacts.
- `scripts/generate_assets.py`: regenerates tables and figures from `outputs/`.
- `Makefile`: one-command build workflow.

## Build

From this directory:

```bash
make all
```

This runs:

```bash
python scripts/generate_assets.py
LC_ALL=C LANG=C latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

## Experiment Figure Templates

The preferred publication-style experiment figure suite is generated with:

```bash
python scripts/generate_topconf_figure_templates.py
```

This writes synthetic source CSV templates to `tables/experiment_source_data_topconf/` and exports editable PDF/SVG, high-resolution TIFF, and PNG preview files to `figures/experiments_topconf/`. It covers primary baselines, legal reliability, convergence and communication, conflict diagnostics, jurisdiction transfer, ablations, privacy trade-offs, and robustness.

After replacing every CSV with verified experimental outputs, render the submission version without placeholder markings:

```bash
python scripts/generate_topconf_figure_templates.py --use-existing-data --final-data
```

See `TOPCONF_FIGURES.md` for the panel-by-panel experimental contract.

The earlier exploratory template script remains available for compatibility:

```bash
python scripts/generate_experiment_figure_templates.py
```

See `EXPERIMENT_FIGURES.md` for the figure-by-figure experiment plan.

## Output Alignment

Generated tables use the current artifacts under:

- `outputs/evaluation/evaluation_default/metrics.json`
- `outputs/logs/aggregation/conflict_aware_fedavg_aggregation.jsonl`
- `outputs/figures/paper/`
- `outputs/figures/conflict_aware_fedavg/`

Current metric values are dry-run diagnostics and should be replaced by full benchmark results before making final empirical claims.
