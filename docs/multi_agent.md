# Federated Multi-Agent Legal Reasoning

The multi-agent system implements Layer 5 from `paper.md`.

## Agents

- `ProsecutorAgent`: accusation-side reasoning
- `DefenseAgent`: rebuttal-side reasoning
- `CitationAgent`: citation extraction and validity checks
- `ConflictAgent`: contradiction, citation divergence, verdict conflict, and jurisdiction incompatibility detection
- `JudgeAgent`: verdict aggregation across debate turns and federated client packets

## Workflow

1. User submits a structured `LegalCase`.
2. Federated clients may provide privacy-preserving `FederatedReasoningPacket` objects.
3. Prosecutor and defense agents produce debate turns.
4. Citation agent verifies cited authorities.
5. Conflict agent analyzes debate and federated packets.
6. Judge agent aggregates arguments, citations, verdict distributions, and conflict flags.
7. Reasoning traces are written as JSONL.

## Federated Setting

Raw local judicial data is not exchanged. Client packets contain only:

- client ID
- jurisdiction
- reasoning embedding
- verdict distribution
- compressed legal representation

## Run

```bash
python scripts/run_reasoning.py configs/experiments/federated_multi_agent_reasoning.yaml case.json
```

Optional packets:

```bash
python scripts/run_reasoning.py configs/experiments/federated_multi_agent_reasoning.yaml case.json --packets packets.json
```

Trace output:

```text
outputs/logs/federated_multi_agent_reasoning_reasoning_traces.jsonl
```
