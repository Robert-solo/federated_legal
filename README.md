# Federated Legal LLMs for Cross-Jurisdiction Judicial Reasoning

This repository implements research workflows for federated legal language modeling, authority-scoped conflict analysis, and multi-agent reasoning. `paper.md` records the original project design.

The maintained manuscript is [the Elsevier CAS source](papers/ipm/Federated%20Legal%20elsevier/cas-sc-template.tex), with its [compiled PDF](papers/ipm/Federated%20Legal%20elsevier/cas-sc-template.pdf). Training runners execute real HuggingFace PEFT/LoRA clients and Flower aggregation. The revised method uses independent shared and private LoRA branches and fixed-budget gating comparisons. A completed 15-run privacy study includes clipping-only controls, record-membership attacks, and probability calibration; the 40-run revised CaseHOLD matrix continues on the server.

See [the revision protocol](experiments/revision_v2/README.md), [human annotation instructions](docs/experiments/HUMAN_ANNOTATION_GUIDE.md), and [the handoff](HANDOFF.md) for commands and current evidence. Expert conflict labels, authentic jurisdiction transfer, and independent legal blind reviews remain pending.

## Architecture Layers

1. Federated judicial data layer: local corpora, citation graphs, legal knowledge bases, and non-IID jurisdiction partitions.
2. Legal LLM local training layer: HuggingFace model loading and PEFT/LoRA adapter configuration.
3. Jurisdiction-aware representation layer: jurisdiction embeddings, legal tradition metadata, and cross-jurisdiction transfer hooks.
4. Conflict-aware aggregation layer: citation, reasoning, verdict, and rule-alignment conflict metrics.
5. Federated multi-agent reasoning layer: LangGraph role skeletons for prosecutor, defense, judge, citation verification, conflict detection, jurisdiction alignment, and privacy auditing.

## Repository Layout

- `src/fedlegal/config`: strongly typed configuration schemas and YAML loading.
- `src/fedlegal/data`: dataset metadata, preprocessing, and non-IID partition planning.
- `src/fedlegal/federated`: Flower client, server, and strategy extension points.
- `src/fedlegal/training`: HuggingFace and PEFT/LoRA pipeline skeleton.
- `src/fedlegal/agents`: LangGraph multi-agent judicial reasoning skeleton.
- `src/fedlegal/aggregation`: conflict-aware aggregation interfaces.
- `src/fedlegal/reasoning`: case decomposition and reasoning orchestration boundaries.
- `src/fedlegal/evaluation`: legal, federated, privacy, and LLM metrics.
- `src/fedlegal/privacy`: differential privacy, secure aggregation, sanitization, and compression hooks.
- `src/fedlegal/retrieval`: future legal RAG and citation graph retrieval hooks.
- `src/fedlegal/figures`: reproducible paper figure generation hooks.
- `configs`: reusable YAML defaults for models, federated simulation, agents, and experiments.
- `experiments`: experiment-specific manifests and reproducibility notes.
- `docs`: architecture and experiment documentation aligned with `paper.md`.

## Quick Start

Install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Validate configuration loading:

```bash
python scripts/validate_config.py configs/experiments/baseline_fedlora.yaml
```

The validation step checks configuration shape only. It does not start training.

Preprocess local legal datasets:

```bash
python scripts/process_datasets.py configs/datasets/legal_supported.yaml --partition
```

The dataset command expects raw local files under `data/raw` and writes normalized records to `data/processed`.

## Research Scope

This is a research-oriented system, not a simplified Legal QA demo. New implementation should preserve:

- federated LoRA communication rather than raw-data centralization
- non-IID cross-jurisdiction simulation
- conflict-aware aggregation
- multi-agent judicial deliberation
- citation consistency and hallucination evaluation
- reproducible experiments with configs, logs, metrics, and figures
