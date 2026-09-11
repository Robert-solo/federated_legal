"""Privacy mechanism planning for federated Legal LLM communication."""

from __future__ import annotations

from dataclasses import dataclass

from fedlegal.config.schemas import PrivacyConfig


@dataclass(frozen=True)
class PrivacyPlan:
    """Enabled privacy mechanisms for one experiment."""

    differential_privacy: bool
    secure_aggregation: bool
    gradient_compression: bool
    sanitize_legal_entities: bool
    noise_multiplier: float


def build_privacy_plan(config: PrivacyConfig) -> PrivacyPlan:
    """Build a privacy plan without mutating model updates."""

    return PrivacyPlan(
        differential_privacy=config.differential_privacy,
        secure_aggregation=config.secure_aggregation,
        gradient_compression=config.gradient_compression,
        sanitize_legal_entities=config.sanitize_legal_entities,
        noise_multiplier=config.noise_multiplier,
    )
