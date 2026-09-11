# AGENTS.md

# Project Overview

This repository implements:

"Federated Legal Large Language Models for Privacy-Preserving Cross-Jurisdiction Judicial Reasoning"

The project focuses on:

- Federated Learning
- Legal LLMs
- Multi-Agent Legal Reasoning
- Conflict-Aware Aggregation
- Cross-Jurisdiction Transfer
- Privacy-Preserving Judicial AI

All implementations must remain aligned with the paper design in `paper.md`.

---

# Global Objectives

The system must support:

1. Federated legal model training
2. Non-IID jurisdiction simulation
3. Conflict-aware aggregation
4. Multi-agent legal reasoning
5. Citation consistency analysis
6. Privacy-preserving communication
7. Reproducible experiments
8. Automatic paper figure generation

---

# Important Constraints

## DO NOT

- rewrite the whole architecture without explicit instruction
- replace the federated pipeline arbitrarily
- remove evaluation metrics
- simplify the legal reasoning pipeline
- ignore cross-jurisdiction heterogeneity

---

# Preferred Technologies

## Federated Learning

- Flower preferred
- FedML optional

---

## LLM Framework

- HuggingFace Transformers
- PEFT
- LoRA
- DeepSpeed

---

## Multi-Agent

- LangGraph preferred
- AutoGen optional

---

# Coding Style

- modular design
- strongly typed configs
- avoid monolithic files
- clear separation between:
  - training
  - reasoning
  - aggregation
  - evaluation

---

# Experiment Rules

Every experiment must include:

- configuration file
- logs
- metrics
- visualization
- reproducibility instructions

---

# Evaluation Requirements

Must include:

- legal accuracy
- citation consistency
- reasoning coherence
- hallucination rate
- federated communication cost
- client drift
- cross-jurisdiction generalization

---

# Documentation Requirements

Every major module must contain:

- README.md
- architecture description
- usage examples

---

# Paper Alignment

All implementations should remain consistent with:

- paper.md
- docs/architecture.md
- docs/experiments.md

---

# Multi-Agent Responsibilities

## Research Agent

Responsible for literature review.

---

## Dataset Agent

Responsible for preprocessing and partitioning.

---

## Federated Agent

Responsible for client-server orchestration.

---

## Legal Reasoning Agent

Responsible for judicial reasoning pipelines.

---

## Conflict Analysis Agent

Responsible for contradiction detection.

---

## Evaluation Agent

Responsible for metrics and plots.

---

# Preferred Workflow

1. Read task file
2. Inspect relevant docs
3. Implement isolated module
4. Add tests
5. Add documentation
6. Run validation
7. Commit clean changes

---

# Important Principle

This repository is a research-oriented system,
not a simple demo project.

All implementations should prioritize:

- extensibility
- reproducibility
- experimental rigor
- paper consistency
