# Architecture

This repository follows the five-layer architecture in `paper.md`.

## Layer 1: Federated Judicial Data

`src/fedlegal/data` owns local dataset metadata and non-IID partition planning. Raw judicial documents remain local to each simulated institution. Later implementations should support CAIL, LeCaRD, ELAM, LJPIV, CaseHOLD, ECtHR, LexGLUE, and CUAD through jurisdiction-aware preprocessing.

## Layer 2: Local Legal LLM Training

`src/fedlegal/training` owns HuggingFace model loading and PEFT/LoRA adapter training. Federated communication should exchange LoRA adapters, gradients, compressed adapters, or reasoning embeddings rather than raw case records. Current compatibility targets are Qwen2.5, Llama3, and Mistral.

## Layer 3: Jurisdiction-Aware Representation

Jurisdiction metadata is encoded in `fedlegal.config.schemas.JurisdictionConfig`. Future work should add jurisdiction embeddings, rule-conditioned routing, legal hierarchy encoders, and cross-jurisdiction alignment losses.

## Layer 4: Target-Conditioned Conflict Routing

`src/fedlegal/aggregation` and `src/fedlegal/federated/strategy.py` define the extension points for citation conflict, reasoning conflict, verdict conflict, and rule-alignment diagnostics. The revised method does not subtract one global conflict scalar from an averaged model. It first checks task and authority comparability, assigns transferable, authority-local, unsupported, or unresolved routing states, and then computes client-specific weights within a target-compatible cohort:

```text
base_kj ∝ authority_compatibility_kj * min(sample_count_k, sample_cap)
candidate_kj ∝ base_kj * exp(-lambda * conflict_exposure_kj)
floored_kj = (1 - tau) * candidate_kj + tau * base_kj
weight_kj(t) = EMA(weight_kj(t-1), floored_kj, smoothing_nu)
```

The diversity floor prevents elimination of a minority client, and unavailable conflict components are omitted with per-pair weight renormalization rather than treated as agreement. The client model composes an uploaded shared LoRA branch with a persistent jurisdiction-local LoRA branch that is never aggregated.

The implementation remains modular:

- `legal_conflict_metrics.py`: conflict report and contradiction penalty
- `citation_divergence.py`: citation inconsistency scoring and citation reliability weights
- `jurisdiction_similarity.py`: rule-alignment similarity and distance
- `aggregators.py`: legacy framework-independent diagnostic aggregator; it must not be used to claim FLEN effectiveness
- planned `target_conditioned.py`: cohort construction, diversity-floored client weights, smoothing, and target-specific aggregation
- planned `flower_flen.py`: Flower runtime strategy and secure-aggregation-compatible weighting

## Layer 5: Federated Multi-Agent Reasoning

`src/fedlegal/agents` defines the LangGraph role plan for jurisdiction alignment, prosecutor, defense, citation verification, conflict detection, privacy audit, and judge aggregation. Future runtime nodes should exchange abstract representations only.

The implemented multi-agent workflow uses `FederatedReasoningPacket` objects for federated-safe exchange, keeps per-case reasoning memory, verifies citations, detects conflicts, and logs auditable reasoning traces.

## Privacy

`src/fedlegal/privacy` centralizes differential privacy, secure aggregation, gradient compression, and legal information sanitization plans. Configuration flags are not evidence that a mechanism executed. A privacy run must log the protected unit, adjacency relation, clipping norm, noise multiplier, sampling rate, accountant, final epsilon/delta, secure-aggregation protocol, dropout/collusion assumptions, and attack results. Probe summaries require a separate privacy treatment from model updates.

## Evaluation

`src/fedlegal/evaluation` records the required metric registry:

- legal accuracy
- citation consistency
- reasoning coherence
- hallucination rate
- communication cost
- client drift
- cross-jurisdiction generalization
- privacy leakage risk
- aggregation stability

The implemented evaluation runner computes citation accuracy, legal consistency,
hallucination rate, communication cost, and client drift, then writes plots,
tables, and reports under `outputs/evaluation`. Authentic cross-jurisdiction claims require a common task or an audited output mapping; results from unrelated datasets must not be averaged or placed in a train-jurisdiction/test-jurisdiction accuracy matrix.

## Figures

`src/fedlegal/figures` is reserved for reproducible generation of paper figures, including the overall framework, aggregation mechanism, multi-agent workflow, privacy communication pipeline, and conflict-aware aggregation architecture.
