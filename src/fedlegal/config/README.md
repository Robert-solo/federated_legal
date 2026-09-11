# Config Module

Typed YAML schemas for reproducible federated legal experiments.

## Architecture

`schemas.py` defines the full experiment contract:

- dataset and jurisdiction metadata
- Flower simulation settings
- HuggingFace and PEFT settings
- conflict-aware aggregation parameters
- privacy controls
- LangGraph agent roles
- evaluation metrics

`loader.py` validates YAML files into `ExperimentConfig`.

## Usage

```python
from fedlegal.config import load_experiment_config

config = load_experiment_config("configs/experiments/baseline_fedlora.yaml")
print(config.federated.strategy)
```
