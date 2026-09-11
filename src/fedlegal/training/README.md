# Training Module

HuggingFace and PEFT/LoRA local adaptation pipeline.

## Architecture

This module corresponds to Layer 2 in `paper.md`. It should own model initialization, tokenizer loading, local supervised fine-tuning, PEFT adapter management, and optional DeepSpeed configuration.

Current scope:

- LoRA adapter planning from typed config
- PEFT and adapter-tuning plan boundaries
- Qwen2.5, Llama3, and Mistral compatibility checks
- local adapter trainer dry-run interface
- model/tokenizer identifiers
- adapter hyperparameter boundaries

Future scope:

- HuggingFace `Trainer` or custom loop integration
- PEFT adapter export/import
- DeepSpeed config generation
- retrieval-augmented fine-tuning hooks

## Usage

```python
from fedlegal.config import load_experiment_config
from fedlegal.training import build_lora_plan

config = load_experiment_config("configs/experiments/baseline_fedlora.yaml")
lora_plan = build_lora_plan(config.model)
```
