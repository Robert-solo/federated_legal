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

The starter-plan scope above applies to the dry runner. The research runtime in
`scripts/run_casehold_revision.py` executes PEFT clients with Flower FedAvg,
FedProx, Ditto, target-conditioned aggregation and independent private adapters.
`adaptive_strategy.CheckpointingFedAdam` adds Flower's persistent-moment server
optimizer and compatible per-round checkpoints/logs. The frozen client and server
settings are in `configs/experiments/revision_v2_extensions.yaml`.

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
