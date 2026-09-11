# Agents Module

LangGraph multi-agent judicial reasoning workflow.

## Architecture

This module corresponds to Layer 5 in `paper.md`.

Configured roles:

- jurisdiction alignment agent
- prosecutor agent
- defense agent
- citation verification agent
- conflict detection agent
- privacy auditor agent
- judge agent

Current scope:

- role graph planning
- exchange-format contract
- deterministic agent implementations
- LangGraph orchestration with fallback execution
- debate workflow
- reasoning memory
- citation verification
- conflict detection
- verdict aggregation
- reasoning trace logging

Future scope:

- LLM-backed role policies
- stronger privacy auditing
- retrieval-grounded citation verification

## Usage

```python
from fedlegal.agents import FederatedLegalReasoningGraph
from fedlegal.config import load_experiment_config
from fedlegal.reasoning import LegalCase

config = load_experiment_config("configs/experiments/federated_multi_agent_reasoning.yaml")
case = LegalCase(case_id="demo", jurisdiction="us_common_law")
state = FederatedLegalReasoningGraph(config.agents, config.conflict).run(case)
```
