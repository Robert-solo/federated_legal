"""Conflict-aware aggregation primitives."""

from fedlegal.aggregation.aggregators import (
    AggregationResult,
    ClientUpdate,
    ConflictAwareFedAvg,
)
from fedlegal.aggregation.citation_divergence import citation_divergence, citation_weight
from fedlegal.aggregation.calibration import (
    ClientRoutingMetadata,
    ConflictCalibrationManifest,
    build_casehold_proxy_manifest,
    load_calibration_manifest,
)
from fedlegal.aggregation.conflict_metrics import ConflictScores, weighted_conflict_penalty
from fedlegal.aggregation.jurisdiction_similarity import (
    jurisdiction_distance,
    jurisdiction_similarity,
)
from fedlegal.aggregation.legal_conflict_metrics import (
    AggregationConflictReport,
    LegalConflictProfile,
    contradiction_penalty,
)
from fedlegal.aggregation.target_conditioned import (
    ClientAdapterBranches,
    ConflictComponents,
    RoutingState,
    TargetAggregationDiagnostics,
    TargetAggregationResult,
    TargetClientUpdate,
    TargetConditionedAggregator,
)
from fedlegal.aggregation.visualization import write_conflict_summary_svg

__all__ = [
    "AggregationConflictReport",
    "AggregationResult",
    "ClientUpdate",
    "ClientAdapterBranches",
    "ClientRoutingMetadata",
    "ConflictCalibrationManifest",
    "ConflictComponents",
    "ConflictAwareFedAvg",
    "ConflictScores",
    "LegalConflictProfile",
    "RoutingState",
    "TargetAggregationDiagnostics",
    "TargetAggregationResult",
    "TargetClientUpdate",
    "TargetConditionedAggregator",
    "citation_divergence",
    "build_casehold_proxy_manifest",
    "citation_weight",
    "contradiction_penalty",
    "jurisdiction_distance",
    "jurisdiction_similarity",
    "load_calibration_manifest",
    "weighted_conflict_penalty",
    "write_conflict_summary_svg",
]
