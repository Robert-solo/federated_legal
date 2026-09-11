# Privacy Module

Privacy-preserving communication controls.

## Architecture

This module corresponds to the privacy mechanisms in `paper.md`:

- differential privacy
- secure aggregation
- federated LoRA
- gradient compression
- legal information sanitization

Current scope:

- privacy plan creation from typed config

Future scope:

- DP noise injection for adapter updates
- secure aggregation protocol wrappers
- legal entity sanitization pipeline
- payload compression accounting

## Usage

```python
from fedlegal.config import load_experiment_config
from fedlegal.privacy import build_privacy_plan

config = load_experiment_config("configs/experiments/conflict_aware.yaml")
privacy_plan = build_privacy_plan(config.privacy)
```
