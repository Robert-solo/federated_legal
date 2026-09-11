"""Flower strategy selection for federated legal LLM training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fedlegal.aggregation.target_conditioned import (
    ConflictComponents,
    TargetClientUpdate,
    TargetConditionedAggregator,
)
from fedlegal.aggregation.calibration import ConflictCalibrationManifest
from fedlegal.config.schemas import ConflictConfig, FederatedConfig

try:
    from flwr.server.strategy import FedAvg as _FlowerFedAvg
except ImportError:
    _FlowerFedAvg = object


@dataclass(frozen=True)
class StrategyPlan:
    """Framework-independent strategy metadata."""

    name: str
    fraction_fit: float
    min_available_clients: int
    fedprox_mu: float | None = None
    scaffold_server_lr: float | None = None
    scaffold_client_lr: float | None = None


def build_strategy_plan(config: FederatedConfig) -> StrategyPlan:
    """Build a strategy plan for supported federated optimizers."""

    if config.strategy not in {"fedavg", "fedprox", "scaffold", "fednova", "conflict_aware"}:
        raise ValueError(f"Unsupported federated strategy: {config.strategy}")
    if config.strategy == "fednova":
        raise NotImplementedError("FedNova is reserved for later baselines.")

    return StrategyPlan(
        name=config.strategy,
        fraction_fit=config.client_fraction,
        min_available_clients=config.min_available_clients,
        fedprox_mu=config.fedprox_mu if config.strategy == "fedprox" else None,
        scaffold_server_lr=config.scaffold_server_lr if config.strategy == "scaffold" else None,
        scaffold_client_lr=config.scaffold_client_lr if config.strategy == "scaffold" else None,
    )


def build_flower_strategy(
    config: FederatedConfig,
    conflict_config: ConflictConfig | None = None,
    calibration_manifest: ConflictCalibrationManifest | None = None,
    *,
    target_jurisdiction: str = "global",
    aggregation_log_path: Path | None = None,
):
    """Build a runtime Flower strategy without conflict-aware fallback."""

    plan = build_strategy_plan(config)
    if plan.name == "conflict_aware" and (
        conflict_config is None or calibration_manifest is None
    ):
        raise ValueError(
            "Conflict-aware Flower aggregation requires explicit conflict and calibration configs; "
            "FedAvg fallback is prohibited."
        )
    try:
        from flwr.server.strategy import FedAvg, FedProx
    except ImportError as exc:
        raise RuntimeError("Flower is required to build runtime server strategies.") from exc

    common = {
        "fraction_fit": plan.fraction_fit,
        "fraction_evaluate": 0.0,
        "min_fit_clients": plan.min_available_clients,
        "min_available_clients": plan.min_available_clients,
    }
    if plan.name == "fedavg":
        return FedAvg(**common)
    if plan.name == "fedprox":
        return FedProx(proximal_mu=plan.fedprox_mu or 0.0, **common)
    if plan.name == "conflict_aware":
        return ConflictAwareFlowerStrategy(
            conflict_config=conflict_config,
            calibration_manifest=calibration_manifest,
            target_jurisdiction=target_jurisdiction,
            aggregation_log_path=aggregation_log_path,
            **common,
        )
    raise NotImplementedError("Scaffold runtime strategy is represented by StrategyPlan only.")


@dataclass(frozen=True)
class ConflictAwareStrategyConfig:
    """Serializable target-conditioned strategy controls."""

    penalty_lambda: float
    citation_weight: float
    reasoning_weight: float
    verdict_weight: float
    rule_alignment_weight: float
    sample_count_cap: int = 10_000
    diversity_floor: float = 0.1
    weight_smoothing: float = 1.0
    target_jurisdiction: str = "global"

    def to_conflict_config(self) -> ConflictConfig:
        """Convert the runtime subset into the typed experiment config."""

        values = asdict(self)
        values.pop("target_jurisdiction")
        return ConflictConfig(**values)


@dataclass(frozen=True)
class PairwiseExposureDiagnostics:
    """Coverage and counts for gold-error-gated probe comparisons."""

    exposures: dict[str, float]
    comparison_counts: dict[str, int]
    expected_comparisons_per_client: int
    prediction_client_coverage: float
    probe_pair_coverage: float


class ConflictAwareFlowerStrategy(_FlowerFedAvg):
    """Flower strategy for one target-conditioned shared-adapter cohort."""

    def __init__(
        self,
        *,
        conflict_config: ConflictConfig,
        calibration_manifest: ConflictCalibrationManifest,
        target_jurisdiction: str,
        probe_labels: tuple[int, ...] | None = None,
        aggregation_log_path: Path | None = None,
        **fedavg_kwargs: Any,
    ) -> None:
        if _FlowerFedAvg is object:
            raise RuntimeError("Flower is required to build ConflictAwareFlowerStrategy.")
        super().__init__(**fedavg_kwargs)
        self.conflict_config = conflict_config
        calibration_manifest.validate()
        if calibration_manifest.target_jurisdiction != target_jurisdiction:
            raise ValueError("Calibration target does not match the Flower strategy target.")
        self.calibration_manifest = calibration_manifest
        self.target_jurisdiction = target_jurisdiction
        self.probe_labels = probe_labels
        self.aggregation_log_path = aggregation_log_path or (
            conflict_config.log_dir / "flower_target_conditioned_aggregation.jsonl"
        )
        self.aggregator = TargetConditionedAggregator(conflict_config)
        self.previous_weights: dict[str, float] | None = None

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate shared adapters and persist a complete per-round audit record."""

        if not results:
            self._write_round_log(server_round, None, results, failures, 0)
            return None, {}
        if failures and not self.accept_failures:
            self._write_round_log(server_round, None, results, failures, 0)
            return None, {}

        from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

        updates: list[TargetClientUpdate] = []
        payload_bytes = 0
        fit_metrics = []
        payloads = []
        predictions: dict[str, tuple[int, ...]] = {}
        for client_proxy, fit_result in results:
            metrics = dict(fit_result.metrics)
            client_id = str(metrics.get("client_id", client_proxy.cid))
            arrays = parameters_to_ndarrays(fit_result.parameters)
            payload_bytes += sum(array.nbytes for array in arrays)
            fit_metrics.append((fit_result.num_examples, metrics))
            if "probe_predictions" in metrics:
                predictions[client_id] = _decode_probe_predictions(
                    metrics["probe_predictions"]
                )
            payloads.append((client_id, arrays, fit_result.num_examples, metrics))

        pairwise = _gold_error_gated_pairwise_diagnostics(
            predictions,
            self.probe_labels,
            expected_client_ids=set(self.calibration_manifest.client_metadata),
        )
        for client_id, arrays, num_examples, metrics in payloads:
            routing = self.calibration_manifest.client(client_id)
            explicit_exposure = _optional_metric(metrics, "conflict_exposure")
            exposure = pairwise.exposures.get(client_id, explicit_exposure)
            updates.append(
                TargetClientUpdate(
                    client_id=client_id,
                    target_jurisdiction=routing.target_jurisdiction,
                    shared_parameters=arrays,
                    num_examples=num_examples,
                    authority_compatibility=routing.authority_compatibility,
                    routing_state=routing.routing_state,
                    conflict_exposure=exposure,
                    conflict_components=(
                        None if exposure is not None else _conflict_components_from_metrics(metrics)
                    ),
                )
            )

        aggregation = self.aggregator.aggregate(updates, self.previous_weights)
        self.previous_weights = aggregation.weights
        metrics_aggregated: dict[str, Any] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated.update(self.fit_metrics_aggregation_fn(fit_metrics))
        available_exposures = [
            exposure
            for exposure in aggregation.diagnostics.conflict_exposures.values()
            if exposure is not None
        ]
        metrics_aggregated.update(
            {
                "cohort_size": len(aggregation.diagnostics.cohort),
                "payload_bytes": payload_bytes,
                "probe_payload_bytes": int(
                    sum(
                        float(dict(fit_result.metrics).get("probe_payload_bytes", 0.0))
                        for _, fit_result in results
                    )
                ),
                "mean_conflict_exposure": (
                    sum(available_exposures) / len(available_exposures)
                    if available_exposures
                    else 0.0
                ),
            }
        )
        self._write_round_log(
            server_round,
            aggregation,
            results,
            failures,
            payload_bytes,
            pairwise,
        )
        return ndarrays_to_parameters(aggregation.shared_parameters), metrics_aggregated

    def _write_round_log(
        self,
        server_round: int,
        aggregation,
        results,
        failures,
        payload_bytes: int,
        pairwise: PairwiseExposureDiagnostics | None = None,
    ) -> None:
        failed_clients = []
        for failure in failures:
            if isinstance(failure, tuple) and failure:
                failed_clients.append(str(getattr(failure[0], "cid", failure[0])))
            else:
                failed_clients.append(str(failure))
        record: dict[str, Any] = {
            "round": server_round,
            "target_jurisdiction": self.target_jurisdiction,
            "num_results": len(results),
            "num_failures": len(failures),
            "failed_clients": failed_clients,
            "payload_bytes": payload_bytes,
            "probe_payload_bytes": int(
                sum(
                    float(dict(fit_result.metrics).get("probe_payload_bytes", 0.0))
                    for _, fit_result in results
                )
            ),
        }
        if aggregation is not None:
            record.update(asdict(aggregation.diagnostics))
        if pairwise is not None:
            record.update(
                {
                    "pairwise_comparison_counts": pairwise.comparison_counts,
                    "expected_comparisons_per_client": (
                        pairwise.expected_comparisons_per_client
                    ),
                    "prediction_client_coverage": pairwise.prediction_client_coverage,
                    "probe_pair_coverage": pairwise.probe_pair_coverage,
                }
            )
        self.aggregation_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.aggregation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _optional_metric(metrics: dict[str, Any], name: str) -> float | None:
    if name not in metrics:
        return None
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Flower metric {name!r} must be finite.")
    return value


def _conflict_components_from_metrics(
    metrics: dict[str, Any],
) -> ConflictComponents | None:
    if "conflict_exposure" in metrics:
        return None
    names = {
        "citation": "citation_conflict",
        "reasoning": "reasoning_conflict",
        "verdict": "verdict_conflict",
        "rule_alignment": "rule_alignment_distance",
    }
    values = {
        component: _optional_metric(metrics, metric_name)
        for component, metric_name in names.items()
    }
    if all(value is None for value in values.values()):
        return None
    return ConflictComponents(**values)


def _decode_probe_predictions(value: Any) -> tuple[int, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("probe_predictions must be a JSON string or bytes scalar.")
    decoded = json.loads(value)
    if not isinstance(decoded, list) or any(not isinstance(item, int) for item in decoded):
        raise ValueError("probe_predictions must encode a list of integer class predictions.")
    return tuple(decoded)


def _gold_error_gated_pairwise_exposures(
    predictions: dict[str, tuple[int, ...]],
    probe_labels: tuple[int, ...] | None,
) -> dict[str, float]:
    return _gold_error_gated_pairwise_diagnostics(predictions, probe_labels).exposures


def _gold_error_gated_pairwise_diagnostics(
    predictions: dict[str, tuple[int, ...]],
    probe_labels: tuple[int, ...] | None,
    *,
    expected_client_ids: set[str] | None = None,
) -> PairwiseExposureDiagnostics:
    if not predictions:
        expected_clients = len(expected_client_ids or ())
        return PairwiseExposureDiagnostics(
            exposures={},
            comparison_counts={},
            expected_comparisons_per_client=0,
            prediction_client_coverage=0.0 if expected_clients else 1.0,
            probe_pair_coverage=0.0 if expected_clients else 1.0,
        )
    if probe_labels is None:
        raise ValueError("Pairwise FLEN exposure requires server-side public probe labels.")
    for client_id, values in predictions.items():
        if len(values) != len(probe_labels):
            raise ValueError(
                f"Probe prediction length mismatch for {client_id!r}: "
                f"{len(values)} != {len(probe_labels)}"
            )
    expected_ids = expected_client_ids or set(predictions)
    unexpected = set(predictions) - expected_ids
    if unexpected:
        raise ValueError(f"Unexpected probe prediction clients: {sorted(unexpected)}")
    expected_per_client = max(len(expected_ids) - 1, 0) * len(probe_labels)
    exposures: dict[str, float] = {}
    comparison_counts: dict[str, int] = {}
    for client_id, client_predictions in predictions.items():
        unsupported = 0
        comparisons = 0
        for peer_id, peer_predictions in predictions.items():
            if peer_id == client_id:
                continue
            for prediction, peer_prediction, label in zip(
                client_predictions,
                peer_predictions,
                probe_labels,
                strict=True,
            ):
                unsupported += int(prediction != peer_prediction and prediction != label)
                comparisons += 1
        exposures[client_id] = unsupported / comparisons if comparisons else 0.0
        comparison_counts[client_id] = comparisons
    total_expected = len(expected_ids) * expected_per_client
    total_observed = sum(comparison_counts.values())
    return PairwiseExposureDiagnostics(
        exposures=exposures,
        comparison_counts=comparison_counts,
        expected_comparisons_per_client=expected_per_client,
        prediction_client_coverage=(
            len(predictions) / len(expected_ids) if expected_ids else 1.0
        ),
        probe_pair_coverage=(
            total_observed / total_expected if total_expected else 1.0
        ),
    )
