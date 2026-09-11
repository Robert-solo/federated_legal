# Federated Module

Flower client-server orchestration for federated Legal LLM adapter training.

## Architecture

This module corresponds to Flower-based coordination in Layers 2 and 4 of `paper.md`.

Current scope:

- FedAvg, FedProx, and Scaffold strategy planning
- Flower NumPy client boundary
- legal client manager
- dry-run server orchestration
- communication logging
- checkpointing
- conflict-aware strategy placeholder only

Future scope:

- LoRA adapter serialization and communication-cost accounting
- full HuggingFace PEFT local training loop
- runtime Scaffold control variate implementation
- Flower `Strategy` subclass for conflict-aware aggregation

## Usage

```python
from fedlegal.config import load_experiment_config
from fedlegal.federated import FlowerServerOrchestrator, build_server_plan

config = load_experiment_config("configs/experiments/fedavg_qwen_lora.yaml")
server_plan = build_server_plan(config)
result = FlowerServerOrchestrator(config).run_dry()
```
