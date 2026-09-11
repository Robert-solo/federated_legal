import math

import numpy as np
import pytest

from fedlegal.aggregation import (
    ClientAdapterBranches,
    ConflictComponents,
    RoutingState,
    TargetClientUpdate,
    TargetConditionedAggregator,
)
from fedlegal.config.schemas import ConflictConfig


def make_update(
    client_id: str,
    value: float,
    *,
    num_examples: int = 10,
    authority_compatibility: float = 1.0,
    routing_state: RoutingState = RoutingState.TRANSFERABLE,
    conflict_exposure: float | None = 0.0,
    conflict_components: ConflictComponents | None = None,
) -> TargetClientUpdate:
    return TargetClientUpdate(
        client_id=client_id,
        target_jurisdiction="target_j",
        shared_parameters=[np.array([value], dtype=np.float64)],
        num_examples=num_examples,
        authority_compatibility=authority_compatibility,
        routing_state=routing_state,
        conflict_exposure=conflict_exposure,
        conflict_components=conflict_components,
    )


def test_lambda_zero_recovers_capped_authority_base_aggregation() -> None:
    config = ConflictConfig(
        penalty_lambda=0.0,
        sample_count_cap=20,
        diversity_floor=0.4,
        weight_smoothing=1.0,
    )
    result = TargetConditionedAggregator(config).aggregate(
        [
            make_update("small", 1.0, num_examples=10, conflict_exposure=1.0),
            make_update(
                "capped",
                3.0,
                num_examples=100,
                authority_compatibility=0.5,
                conflict_exposure=0.0,
            ),
        ]
    )

    assert result.weights == pytest.approx({"capped": 0.5, "small": 0.5})
    assert result.shared_parameters[0] == pytest.approx(np.array([2.0]))


def test_equal_exposure_preserves_base_weight_ratio() -> None:
    config = ConflictConfig(penalty_lambda=0.8, diversity_floor=0.2)
    result = TargetConditionedAggregator(config).aggregate(
        [
            make_update("major", 1.0, num_examples=30, conflict_exposure=0.6),
            make_update("minor", 3.0, num_examples=10, conflict_exposure=0.6),
        ]
    )

    assert result.weights["major"] / result.weights["minor"] == pytest.approx(3.0)


def test_increasing_one_exposure_cannot_increase_candidate_weight() -> None:
    config = ConflictConfig(penalty_lambda=0.8, diversity_floor=0.1)
    aggregator = TargetConditionedAggregator(config)
    low = aggregator.aggregate(
        [make_update("a", 1.0, conflict_exposure=0.1), make_update("b", 2.0)]
    )
    high = aggregator.aggregate(
        [make_update("a", 1.0, conflict_exposure=0.9), make_update("b", 2.0)]
    )

    assert (
        high.diagnostics.candidate_weights["a"]
        <= low.diagnostics.candidate_weights["a"]
    )


def test_diversity_floor_retains_declared_base_mass() -> None:
    config = ConflictConfig(penalty_lambda=20.0, diversity_floor=0.25)
    result = TargetConditionedAggregator(config).aggregate(
        [make_update("minor", 1.0, conflict_exposure=1.0), make_update("major", 2.0)]
    )

    floor = config.diversity_floor * result.diagnostics.base_weights["minor"]
    assert result.diagnostics.floored_weights["minor"] >= floor


def test_authority_incompatible_and_authority_local_clients_are_excluded() -> None:
    result = TargetConditionedAggregator(ConflictConfig()).aggregate(
        [
            make_update("included", 1.0),
            make_update("incompatible", 100.0, authority_compatibility=0.0),
            make_update(
                "local",
                100.0,
                routing_state=RoutingState.AUTHORITY_LOCAL,
            ),
        ]
    )

    assert result.weights == {"included": 1.0}
    assert result.diagnostics.excluded_clients == {
        "incompatible": "authority_incompatible",
        "local": "authority_local",
    }


def test_weights_are_finite_non_negative_normalized_and_order_invariant() -> None:
    config = ConflictConfig(penalty_lambda=0.4, diversity_floor=0.15, weight_smoothing=0.6)
    updates = [
        make_update("c", 3.0, num_examples=7, conflict_exposure=0.7),
        make_update("a", 1.0, num_examples=13, conflict_exposure=0.1),
        make_update("b", 2.0, num_examples=5, conflict_exposure=0.4),
    ]
    aggregator = TargetConditionedAggregator(config)
    forward = aggregator.aggregate(updates, previous_weights={"a": 0.2, "b": 0.3, "c": 0.5})
    reverse = aggregator.aggregate(
        list(reversed(updates)), previous_weights={"a": 0.2, "b": 0.3, "c": 0.5}
    )

    assert forward.weights == pytest.approx(reverse.weights)
    assert forward.shared_parameters[0] == pytest.approx(reverse.shared_parameters[0])
    assert sum(forward.weights.values()) == pytest.approx(1.0)
    assert all(math.isfinite(weight) and weight >= 0 for weight in forward.weights.values())


def test_majority_conflict_does_not_globally_suppress_valid_minority() -> None:
    config = ConflictConfig(penalty_lambda=2.0, diversity_floor=0.2)
    result = TargetConditionedAggregator(config).aggregate(
        [
            make_update("majority_1", 1.0, num_examples=100, conflict_exposure=0.9),
            make_update("majority_2", 1.0, num_examples=100, conflict_exposure=0.9),
            make_update("minority", 5.0, num_examples=10, conflict_exposure=0.0),
        ]
    )

    base = result.diagnostics.base_weights
    candidate = result.diagnostics.candidate_weights
    assert candidate["minority"] > base["minority"]
    assert candidate["majority_1"] < base["majority_1"]


def test_missing_components_are_renormalized_over_available_components() -> None:
    config = ConflictConfig(
        citation_weight=0.25,
        reasoning_weight=0.5,
        verdict_weight=0.25,
        rule_alignment_weight=0.0,
    )
    update = make_update(
        "partial",
        1.0,
        conflict_exposure=None,
        conflict_components=ConflictComponents(citation=0.2, reasoning=0.8),
    )
    result = TargetConditionedAggregator(config).aggregate([update])

    expected = (0.25 * 0.2 + 0.5 * 0.8) / 0.75
    assert result.diagnostics.conflict_exposures["partial"] == pytest.approx(expected)
    assert result.diagnostics.component_denominators["partial"] == pytest.approx(0.75)


def test_empty_cohort_raises_and_dropout_renormalizes_current_clients() -> None:
    aggregator = TargetConditionedAggregator(ConflictConfig(weight_smoothing=0.5))
    with pytest.raises(ValueError, match="No eligible clients"):
        aggregator.aggregate([make_update("local", 1.0, authority_compatibility=0.0)])

    result = aggregator.aggregate(
        [make_update("remaining", 2.0)],
        previous_weights={"remaining": 0.2, "dropped": 0.8},
    )
    assert result.weights == {"remaining": 1.0}


def test_local_residual_is_not_part_of_server_upload_or_aggregation() -> None:
    branches = ClientAdapterBranches(
        client_id="client",
        shared_parameters=[np.array([2.0])],
        local_parameters=[np.array([99.0])],
    )
    upload = branches.shared_upload(
        target_jurisdiction="target_j",
        num_examples=10,
        authority_compatibility=1.0,
        routing_state=RoutingState.TRANSFERABLE,
        conflict_exposure=0.0,
    )
    result = TargetConditionedAggregator(ConflictConfig()).aggregate([upload])

    assert not hasattr(upload, "local_parameters")
    assert result.shared_parameters[0] == pytest.approx(np.array([2.0]))
    assert branches.local_parameters[0] == pytest.approx(np.array([99.0]))
