"""Flower-based federated learning orchestration."""

from fedlegal.federated.client import LegalFlowerClient
from fedlegal.federated.client_manager import ClientDescriptor, LegalClientManager
from fedlegal.federated.logging import CommunicationEvent, CommunicationLogger
from fedlegal.federated.server import FlowerServerOrchestrator, OrchestrationResult, build_server_plan
from fedlegal.federated.strategy import (
    ConflictAwareFlowerStrategy,
    ConflictAwareStrategyConfig,
    StrategyPlan,
    build_flower_strategy,
    build_strategy_plan,
)

__all__ = [
    "ClientDescriptor",
    "CommunicationEvent",
    "CommunicationLogger",
    "ConflictAwareStrategyConfig",
    "ConflictAwareFlowerStrategy",
    "FlowerServerOrchestrator",
    "LegalClientManager",
    "LegalFlowerClient",
    "OrchestrationResult",
    "StrategyPlan",
    "build_flower_strategy",
    "build_server_plan",
    "build_strategy_plan",
]
