"""Target-conditioned aggregation for conflict-aware federated legal adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fedlegal.config.schemas import ConflictConfig


class RoutingState(str, Enum):
    """Legally typed transfer decision for one client-to-target update."""

    TRANSFERABLE = "transferable"
    AUTHORITY_LOCAL = "authority_local"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ConflictComponents:
    """Available conflict components for one client-to-target exposure."""

    citation: float | None = None
    reasoning: float | None = None
    verdict: float | None = None
    rule_alignment: float | None = None

    def weighted_exposure(self, config: ConflictConfig) -> tuple[float | None, float]:
        """Return the available-component weighted exposure and denominator."""

        components = (
            (self.citation, config.citation_weight),
            (self.reasoning, config.reasoning_weight),
            (self.verdict, config.verdict_weight),
            (self.rule_alignment, config.rule_alignment_weight),
        )
        available: list[tuple[float, float]] = []
        for value, weight in components:
            if value is None:
                continue
            _validate_unit_interval(value, "conflict component")
            if weight > 0:
                available.append((value, weight))
        denominator = sum(weight for _, weight in available)
        if denominator <= 0:
            return None, 0.0
        exposure = sum(value * weight for value, weight in available) / denominator
        return exposure, denominator


@dataclass(frozen=True)
class TargetClientUpdate:
    """Uploaded shared-adapter update and target-scoped legal metadata."""

    client_id: str
    target_jurisdiction: str
    shared_parameters: list[Any]
    num_examples: int
    authority_compatibility: float
    routing_state: RoutingState
    conflict_exposure: float | None = None
    conflict_components: ConflictComponents | None = None


@dataclass(frozen=True)
class ClientAdapterBranches:
    """Client-side adapter state with a non-uploadable local residual."""

    client_id: str
    shared_parameters: list[Any]
    local_parameters: list[Any]

    def shared_upload(
        self,
        *,
        target_jurisdiction: str,
        num_examples: int,
        authority_compatibility: float,
        routing_state: RoutingState,
        conflict_exposure: float | None = None,
        conflict_components: ConflictComponents | None = None,
    ) -> TargetClientUpdate:
        """Create the server payload without serializing the local branch."""

        return TargetClientUpdate(
            client_id=self.client_id,
            target_jurisdiction=target_jurisdiction,
            shared_parameters=self.shared_parameters,
            num_examples=num_examples,
            authority_compatibility=authority_compatibility,
            routing_state=routing_state,
            conflict_exposure=conflict_exposure,
            conflict_components=conflict_components,
        )


@dataclass(frozen=True)
class TargetAggregationDiagnostics:
    """Auditable intermediate values for one target aggregation round."""

    target_jurisdiction: str
    cohort: tuple[str, ...]
    routing_states: dict[str, str]
    excluded_clients: dict[str, str]
    conflict_exposures: dict[str, float | None]
    component_denominators: dict[str, float]
    base_masses: dict[str, float]
    base_weights: dict[str, float]
    candidate_weights: dict[str, float]
    floored_weights: dict[str, float]
    smoothed_weights: dict[str, float]
    final_weights: dict[str, float]


@dataclass(frozen=True)
class TargetAggregationResult:
    """Target-specific shared adapter and its aggregation audit trail."""

    target_jurisdiction: str
    shared_parameters: list[Any]
    weights: dict[str, float]
    diagnostics: TargetAggregationDiagnostics


class TargetConditionedAggregator:
    """Aggregate one target cohort using client-specific conflict exposure."""

    def __init__(self, config: ConflictConfig) -> None:
        self.config = config

    def aggregate(
        self,
        updates: list[TargetClientUpdate],
        previous_weights: dict[str, float] | None = None,
    ) -> TargetAggregationResult:
        """Aggregate uploaded shared branches for exactly one target jurisdiction."""

        if not updates:
            raise ValueError("At least one target client update is required.")
        ordered = sorted(updates, key=lambda update: update.client_id)
        self._validate_updates(ordered)
        target_jurisdiction = ordered[0].target_jurisdiction
        cohort, excluded, exposures, denominators = self._build_cohort(ordered)
        if not cohort:
            raise ValueError(f"No eligible clients for target {target_jurisdiction!r}.")

        base_masses = {
            update.client_id: update.authority_compatibility
            * min(max(update.num_examples, 0), self.config.sample_count_cap)
            for update in cohort
        }
        base_weights = _normalize(base_masses, "base masses")
        candidate_masses = {
            client_id: base_weights[client_id]
            * math.exp(-self.config.penalty_lambda * (exposures[client_id] or 0.0))
            for client_id in base_weights
        }
        candidate_weights = _normalize(candidate_masses, "candidate masses")
        floored_weights = {
            client_id: (1.0 - self.config.diversity_floor) * candidate_weights[client_id]
            + self.config.diversity_floor * base_weights[client_id]
            for client_id in base_weights
        }
        floored_weights = _normalize(floored_weights, "floored weights")
        prior = self._current_cohort_prior(previous_weights, base_weights)
        smoothed_weights = {
            client_id: (1.0 - self.config.weight_smoothing) * prior[client_id]
            + self.config.weight_smoothing * floored_weights[client_id]
            for client_id in base_weights
        }
        final_weights = _normalize(smoothed_weights, "smoothed weights")
        shared_parameters = _aggregate_parameters(cohort, final_weights)
        diagnostics = TargetAggregationDiagnostics(
            target_jurisdiction=target_jurisdiction,
            cohort=tuple(update.client_id for update in cohort),
            routing_states={
                update.client_id: update.routing_state.value for update in ordered
            },
            excluded_clients=excluded,
            conflict_exposures=exposures,
            component_denominators=denominators,
            base_masses=base_masses,
            base_weights=base_weights,
            candidate_weights=candidate_weights,
            floored_weights=floored_weights,
            smoothed_weights=smoothed_weights,
            final_weights=final_weights,
        )
        return TargetAggregationResult(
            target_jurisdiction=target_jurisdiction,
            shared_parameters=shared_parameters,
            weights=final_weights,
            diagnostics=diagnostics,
        )

    def _validate_updates(self, updates: list[TargetClientUpdate]) -> None:
        client_ids = [update.client_id for update in updates]
        if len(client_ids) != len(set(client_ids)):
            raise ValueError("Client identifiers must be unique within a target cohort.")
        targets = {update.target_jurisdiction for update in updates}
        if len(targets) != 1:
            raise ValueError("A target-conditioned aggregation call must contain exactly one target.")
        for update in updates:
            _validate_unit_interval(update.authority_compatibility, "authority compatibility")
            if update.num_examples < 0:
                raise ValueError("Client sample counts cannot be negative.")
            if update.conflict_exposure is not None:
                _validate_unit_interval(update.conflict_exposure, "conflict exposure")
            if update.conflict_exposure is not None and update.conflict_components is not None:
                raise ValueError(
                    "Provide either a calibrated conflict exposure or conflict components, not both."
                )

    def _build_cohort(
        self,
        updates: list[TargetClientUpdate],
    ) -> tuple[
        list[TargetClientUpdate],
        dict[str, str],
        dict[str, float | None],
        dict[str, float],
    ]:
        cohort: list[TargetClientUpdate] = []
        excluded: dict[str, str] = {}
        exposures: dict[str, float | None] = {}
        denominators: dict[str, float] = {}
        for update in updates:
            if update.authority_compatibility <= 0:
                excluded[update.client_id] = "authority_incompatible"
                continue
            if update.routing_state is RoutingState.AUTHORITY_LOCAL:
                excluded[update.client_id] = RoutingState.AUTHORITY_LOCAL.value
                continue
            if (
                self.config.cohort_policy == "transferable_only"
                and update.routing_state is not RoutingState.TRANSFERABLE
            ):
                excluded[update.client_id] = "cohort_policy"
                continue
            exposure, denominator = self._resolve_exposure(update)
            if exposure is None and self.config.missing_component_policy == "exclude_client":
                excluded[update.client_id] = "missing_conflict_evidence"
                continue
            if exposure is None and self.config.missing_component_policy == "error":
                raise ValueError(f"Client {update.client_id!r} has no conflict evidence.")
            cohort.append(update)
            exposures[update.client_id] = exposure
            denominators[update.client_id] = denominator
        return cohort, excluded, exposures, denominators

    def _resolve_exposure(
        self,
        update: TargetClientUpdate,
    ) -> tuple[float | None, float]:
        if update.conflict_exposure is not None:
            return update.conflict_exposure, 1.0
        if update.conflict_components is None:
            return None, 0.0
        return update.conflict_components.weighted_exposure(self.config)

    def _current_cohort_prior(
        self,
        previous_weights: dict[str, float] | None,
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        if previous_weights is None:
            return dict(base_weights)
        prior = {}
        for client_id, base_weight in base_weights.items():
            value = previous_weights.get(client_id, base_weight)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Previous weights must be finite and non-negative.")
            prior[client_id] = value
        if sum(prior.values()) <= 0:
            return dict(base_weights)
        return _normalize(prior, "previous weights")


def _validate_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name.capitalize()} must be finite and lie in [0, 1].")


def _normalize(values: dict[str, float], name: str) -> dict[str, float]:
    if not values:
        raise ValueError(f"Cannot normalize empty {name}.")
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"{name.capitalize()} must be finite and non-negative.")
    denominator = sum(values.values())
    if denominator <= 0:
        raise ValueError(f"{name.capitalize()} must contain positive mass.")
    return {client_id: values[client_id] / denominator for client_id in sorted(values)}


def _aggregate_parameters(
    updates: list[TargetClientUpdate],
    weights: dict[str, float],
) -> list[Any]:
    parameter_groups = [update.shared_parameters for update in updates]
    if not parameter_groups or not parameter_groups[0]:
        raise ValueError("Shared adapter parameters cannot be empty.")
    expected_groups = len(parameter_groups[0])
    if any(len(parameters) != expected_groups for parameters in parameter_groups):
        raise ValueError("All shared adapter updates must have the same parameter structure.")
    ordered_weights = [weights[update.client_id] for update in updates]
    return [
        _weighted_sum(list(values), ordered_weights)
        for values in zip(*parameter_groups, strict=True)
    ]


def _weighted_sum(values: list[Any], weights: list[float]) -> Any:
    first = values[0]
    if isinstance(first, list):
        if any(len(value) != len(first) for value in values):
            raise ValueError("Nested parameter lists must have matching lengths.")
        return [
            _weighted_sum([value[index] for value in values], weights)
            for index in range(len(first))
        ]
    if hasattr(first, "__mul__") and not isinstance(first, (str, bytes)):
        result = first * weights[0]
        for weight, value in zip(weights[1:], values[1:], strict=True):
            result = result + value * weight
        return result
    return sum(weight * float(value) for weight, value in zip(weights, values, strict=True))
