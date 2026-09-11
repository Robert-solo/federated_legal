# Federated Learning

This project uses Flower as the preferred federated learning framework for legal LLM adapter federation.

## Scope

Implemented now:

- FedAvg strategy planning
- FedProx strategy planning with `fedprox_mu`
- Scaffold strategy planning with server/client learning-rate metadata
- Flower NumPy client adapter boundary
- local adapter trainer boundary for LoRA, PEFT, and adapter tuning
- client manager for jurisdiction and institution metadata
- communication logging
- round checkpointing
- deterministic dry-run orchestration

Implemented for conflict-aware aggregation:

- framework-independent conflict-aware FedAvg
- citation-weighted aggregation
- contradiction penalty
- aggregation logs
- SVG diagnostics

Not implemented yet:

- full HuggingFace gradient training
- runtime Scaffold control variate updates
- secure aggregation protocol internals

## Strategies

### FedAvg

Baseline aggregation over client adapter payloads.

### FedProx

Adds a proximal coefficient to reduce local drift under non-IID jurisdiction partitions.

### Scaffold

Represented as an explicit strategy plan. Runtime control variate updates will be implemented later, but configs and orchestration metadata are already supported.

## Adapter Support

The training boundary supports:

- `lora`
- `peft`
- `adapter`
- `adapter_tuning`

Compatible model families:

- Qwen2.5
- Llama3
- Mistral

## Dry Run

Dry run validates orchestration without model downloads:

```bash
python scripts/run_federated.py configs/experiments/fedavg_qwen_lora.yaml
```

Outputs:

```text
outputs/checkpoints/<run_name>/round_0001.npz
outputs/checkpoints/<run_name>/round_0001.json
outputs/logs/communication/<run_name>.jsonl
```

## Runtime Flower

The runtime client and strategy adapters are wired, but full LLM training is intentionally guarded until the local HuggingFace training loop is implemented:

```bash
python scripts/run_federated.py configs/experiments/fedavg_qwen_lora.yaml --runtime
```

## Conflict-Aware Aggregation

Conflict-aware aggregation follows the paper objective:

```text
w_{t+1} = sum_k alpha_k w_k - lambda * D_conflict
```

The implementation applies the conflict penalty as bounded shrinkage of client weights before FedAvg normalization. This keeps adapter tensor types stable while reducing the influence of legally contradictory updates.

Run:

```bash
python scripts/run_federated.py configs/experiments/conflict_aware_fedavg.yaml
```

Diagnostics:

```text
outputs/logs/aggregation/conflict_aware_fedavg_aggregation.jsonl
outputs/figures/conflict_aware_fedavg/conflict_round_0001.svg
```
