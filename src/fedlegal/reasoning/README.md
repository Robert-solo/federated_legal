# Reasoning Module

Case decomposition and judicial reasoning interfaces.

## Architecture

This module connects user-submitted cases to the federated multi-agent reasoning workflow described in `paper.md`.

Current scope:

- typed `LegalCase` schema

Future scope:

- fact and issue decomposition
- jurisdiction-specific reasoning prompt builders
- verdict distribution schemas
- reasoning-chain validation

## Usage

```python
from fedlegal.reasoning import LegalCase

case = LegalCase(case_id="demo-001", jurisdiction="us_common_law")
```
