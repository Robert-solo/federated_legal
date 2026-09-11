# Aggregation Module

Conflict-aware aggregation for cross-jurisdiction federated legal models.

## Architecture

This module corresponds to Layer 4 in `paper.md`.

Conflict dimensions:

- citation conflict
- reasoning conflict
- verdict conflict
- rule-alignment distance

Current scope:

- conflict score data structure
- weighted conflict penalty calculation
- citation divergence
- jurisdiction similarity and rule-alignment distance
- conflict-aware FedAvg
- citation-weighted aggregation
- contradiction penalty
- aggregation logging
- dependency-free SVG diagnostics

Future scope:

- contradiction detection model integration
- precedent-family alignment scoring
- Flower strategy integration

## Usage

```python
from fedlegal.aggregation import ClientUpdate, ConflictAwareFedAvg, LegalConflictProfile
from fedlegal.config import load_experiment_config

config = load_experiment_config("configs/experiments/conflict_aware_fedavg.yaml")
aggregator = ConflictAwareFedAvg(config.conflict)
result = aggregator.aggregate([
    ClientUpdate(
        client_id="court_us_01",
        parameters=[[1.0]],
        num_examples=10,
        profile=LegalConflictProfile(
            client_id="court_us_01",
            citations=("42 U.S.C. § 1983",),
            jurisdiction="us_common_law",
            legal_tradition="common_law",
        ),
    )
])
```
