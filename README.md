# Federated Legal LLMs for Cross-Jurisdiction Judicial Reasoning

This repository scaffolds the research system described in `paper.md`: a privacy-preserving federated Legal LLM framework for cross-jurisdiction judicial reasoning.

The current state is architecture-only. It defines package boundaries, typed configuration schemas, placeholder experiment configs, and integration points for Flower, HuggingFace PEFT/LoRA, and LangGraph. Full training and production reasoning are intentionally not implemented yet.

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
