"""YAML configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fedlegal.config.schemas import DataConfig, ExperimentConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration."""

    return ExperimentConfig.model_validate(load_yaml(path))


def load_data_config(path: str | Path) -> DataConfig:
    """Load a standalone data config or the `data` section of an experiment config."""

    payload = load_yaml(path)
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping for data config in {path}")
    return DataConfig.model_validate(data)
