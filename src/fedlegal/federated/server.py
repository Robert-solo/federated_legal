"""Flower server planning and orchestration for federated legal simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedlegal.aggregation import (
    ClientUpdate,
    ConflictAwareFedAvg,
    LegalConflictProfile,
    write_conflict_summary_svg,
)
from fedlegal.aggregation.logging import AggregationLogger
from fedlegal.config.schemas import ExperimentConfig
from fedlegal.federated.checkpointing import CheckpointManager
from fedlegal.federated.client import LegalFlowerClient
from fedlegal.federated.client_manager import LegalClientManager
from fedlegal.federated.logging import CommunicationEvent, CommunicationLogger
from fedlegal.federated.strategy import StrategyPlan, build_strategy_plan
from fedlegal.training import LocalAdapterTrainer, build_adapter_tuning_plan


@dataclass(frozen=True)
class ServerPlan:
    """Execution plan for a Flower simulation."""

    strategy: str
    num_rounds: int
    num_clients: int
    min_available_clients: int
    payload: str
    checkpoint_dir: str
    communication_log_dir: str
    strategy_plan: StrategyPlan


def build_server_plan(config: ExperimentConfig) -> ServerPlan:
    """Create a server plan without starting a Flower simulation."""

    fed = config.federated
    return ServerPlan(
        strategy=fed.strategy,
        num_rounds=fed.num_rounds,
        num_clients=fed.num_clients,
        min_available_clients=fed.min_available_clients,
        payload=fed.communication_payload,
        checkpoint_dir=str(fed.checkpoint_dir),
        communication_log_dir=str(fed.communication_log_dir),
        strategy_plan=build_strategy_plan(fed),
    )


@dataclass
class OrchestrationResult:
    """Summary returned after a dry-run or runtime server orchestration."""

    run_name: str
    strategy: str
    rounds: int
    clients: list[str]
    checkpoint_dir: str
    communication_log: str
    metrics: dict[str, Any]


class FlowerServerOrchestrator:
    """Federated legal LLM server orchestration.

    `dry_run=True` exercises client scheduling, communication logging, and
    checkpointing with deterministic adapter payloads. Full Flower simulation is
    available as an extension point once real local training is implemented.
    """

    def __init__(self, config: ExperimentConfig, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self.plan = build_server_plan(config)
        self.client_manager = LegalClientManager.from_data_config(config.data)
        self.communication_logger = CommunicationLogger(
            config.federated.communication_log_dir,
            config.name,
        )
        self.checkpoints = CheckpointManager(config.federated.checkpoint_dir, config.name)
        self.aggregation_logger = AggregationLogger(config.conflict.log_dir, config.name)

    def build_clients(self) -> list[LegalFlowerClient]:
        """Create legal Flower clients from jurisdiction descriptors."""

        adapter_plan = build_adapter_tuning_plan(self.config.model)
        clients: list[LegalFlowerClient] = []
        for descriptor in self.client_manager.all():
            trainer = LocalAdapterTrainer(
                adapter_plan,
                dataset_path=descriptor.partition_path,
                dry_run=self.dry_run,
                jurisdiction=descriptor.jurisdiction,
                legal_tradition=descriptor.legal_tradition,
            )
            clients.append(
                LegalFlowerClient(
                    client_id=descriptor.client_id,
                    jurisdiction=descriptor.jurisdiction,
                    trainer=trainer,
                    local_epochs=self.config.federated.local_epochs,
                    communication_logger=self.communication_logger,
                    payload_type=self.config.federated.communication_payload,
                )
            )
        return clients

    def run_dry(self) -> OrchestrationResult:
        """Run deterministic local orchestration without starting a Flower server."""

        clients = self.build_clients()
        if not clients:
            raise ValueError("No clients registered for federated orchestration.")

        global_parameters = clients[0].get_adapter_parameters()
        parameter_keys = clients[0].parameter_keys
        round_metrics: dict[str, Any] = {}
        for round_id in range(1, self.config.federated.num_rounds + 1):
            client_results = [
                client.fit_local_adapter(parameters=global_parameters, round_id=round_id)
                for client in clients
            ]
            if self.config.federated.strategy == "conflict_aware" and self.config.conflict.enabled:
                aggregation_result = ConflictAwareFedAvg(
                    self.config.conflict,
                    citation_weighted=self.config.conflict.citation_weighted_aggregation,
                    contradiction_penalty_enabled=self.config.conflict.contradiction_penalty,
                ).aggregate(
                    [
                        ClientUpdate(
                            client_id=client.client_id,
                            parameters=parameters,
                            num_examples=num_examples,
                            profile=_profile_for_client(client, metrics),
                        )
                        for client, (parameters, num_examples, metrics) in zip(
                            clients,
                            client_results,
                            strict=True,
                        )
                    ]
                )
                global_parameters = aggregation_result.parameters
                self.aggregation_logger.log_round(round_id, aggregation_result)
                if self.config.conflict.visualization:
                    write_conflict_summary_svg(
                        aggregation_result.conflict_report,
                        self.config.figure_dir
                        / self.config.name
                        / f"conflict_round_{round_id:04d}.svg",
                    )
                aggregation_metrics = {
                    "total_conflict": aggregation_result.conflict_report.total_conflict,
                    "citation_conflict": aggregation_result.conflict_report.citation_conflict,
                    "contradiction_penalty": aggregation_result.conflict_report.contradiction_penalty,
                }
            else:
                global_parameters = _fedavg_parameters(client_results)
                aggregation_metrics = {}
            total_examples = sum(num_examples for _, num_examples, _ in client_results)
            round_metrics = {
                "total_examples": total_examples,
                "num_clients": len(clients),
                **aggregation_metrics,
            }
            self.checkpoints.save_round(
                round_id,
                parameter_keys,
                global_parameters,
                metrics=round_metrics,
            )
            self.communication_logger.log(
                CommunicationEvent(
                    round_id=round_id,
                    client_id="server",
                    direction="server_checkpoint",
                    payload_type=self.config.federated.communication_payload,
                    num_tensors=len(global_parameters),
                    num_bytes=sum(_parameter_nbytes(parameter) for parameter in global_parameters),
                    metrics=round_metrics,
                )
            )

        return OrchestrationResult(
            run_name=self.config.name,
            strategy=self.config.federated.strategy,
            rounds=self.config.federated.num_rounds,
            clients=[client.client_id for client in clients],
            checkpoint_dir=str(self.checkpoints.run_dir),
            communication_log=str(self.communication_logger.path),
            metrics=round_metrics,
        )

    def run_flower_simulation(self) -> OrchestrationResult:
        """Start a Flower simulation.

        This method is intentionally guarded because real LLM training and
        environment-specific Flower setup are not part of this implementation.
        """

        raise NotImplementedError(
            "Flower runtime simulation is wired through client/strategy adapters, "
            "but full LLM training is not implemented yet."
        )


def _fedavg_parameters(client_results):
    total_examples = sum(num_examples for _, num_examples, _ in client_results)
    if total_examples <= 0:
        total_examples = len(client_results)
        weights = [1.0 / total_examples] * len(client_results)
    else:
        weights = [num_examples / total_examples for _, num_examples, _ in client_results]

    parameters_by_client = [parameters for parameters, _, _ in client_results]
    aggregated = []
    for parameter_group in zip(*parameters_by_client, strict=True):
        value = _weighted_sum(parameter_group, weights)
        aggregated.append(value)
    return aggregated


def _weighted_sum(values, weights):
    first = values[0]
    if hasattr(first, "__mul__") and not isinstance(first, list):
        return sum(weight * value for weight, value in zip(weights, values, strict=True))
    if isinstance(first, list):
        return [
            _weighted_sum([value[index] for value in values], weights)
            for index in range(len(first))
        ]
    return sum(weight * float(value) for weight, value in zip(weights, values, strict=True))


def _parameter_nbytes(parameter) -> int:
    nbytes = getattr(parameter, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return len(str(parameter).encode("utf-8"))


def _profile_for_client(client: LegalFlowerClient, metrics: dict[str, Any]) -> LegalConflictProfile:
    """Build legal conflict profile from client metadata and optional metrics."""

    return LegalConflictProfile(
        client_id=client.client_id,
        citations=tuple(str(item) for item in metrics.get("citations", ())),
        reasoning_embedding=tuple(float(item) for item in metrics.get("reasoning_embedding", ())),
        verdict_distribution=tuple(float(item) for item in metrics.get("verdict_distribution", ())),
        jurisdiction=str(metrics.get("jurisdiction", client.jurisdiction)),
        legal_tradition=str(metrics.get("legal_tradition", "")),
        contradiction_score=float(metrics.get("contradiction_score", 0.0)),
    )
