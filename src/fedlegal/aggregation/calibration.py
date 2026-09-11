"""Typed calibration manifests for target-conditioned conflict routing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fedlegal.aggregation.target_conditioned import RoutingState


EvidenceLevel = Literal["automatic_proxy_pilot", "expert_adjudicated"]


@dataclass(frozen=True)
class ClientRoutingMetadata:
    """Frozen authority and routing metadata for one client and target."""

    client_id: str
    target_jurisdiction: str
    authority_compatibility: float
    routing_state: RoutingState

    def validate(self) -> None:
        if not self.client_id:
            raise ValueError("Calibration client_id cannot be empty.")
        if not self.target_jurisdiction:
            raise ValueError("Calibration target_jurisdiction cannot be empty.")
        if not 0.0 <= self.authority_compatibility <= 1.0:
            raise ValueError("Authority compatibility must lie in [0, 1].")


@dataclass(frozen=True)
class ConflictCalibrationManifest:
    """Auditable calibration contract consumed by the Flower runtime."""

    schema_version: str
    calibration_id: str
    evidence_level: EvidenceLevel
    formal_legal_claims_allowed: bool
    task: str
    target_jurisdiction: str
    probe_path: str
    probe_records: int
    exposure_component: Literal["unsupported_pairwise_verdict"]
    exposure_transform: Literal["gold_error_gated_disagreement"]
    component_weights: dict[str, float]
    client_metadata: dict[str, ClientRoutingMetadata]
    annotation_summary: dict[str, int]
    limitations: tuple[str, ...]

    def validate(self, *, require_expert: bool = False) -> None:
        if not self.calibration_id or not self.task or not self.target_jurisdiction:
            raise ValueError("Calibration identifiers, task, and target are required.")
        if self.evidence_level not in {"automatic_proxy_pilot", "expert_adjudicated"}:
            raise ValueError(f"Unsupported calibration evidence level: {self.evidence_level!r}.")
        if self.probe_records <= 0:
            raise ValueError("Calibration probe must contain at least one record.")
        if not self.client_metadata:
            raise ValueError("Calibration manifest must contain client metadata.")
        for client_id, metadata in self.client_metadata.items():
            metadata.validate()
            if client_id != metadata.client_id:
                raise ValueError("Calibration client_metadata keys must match client_id values.")
            if metadata.target_jurisdiction != self.target_jurisdiction:
                raise ValueError("All pilot clients must reference the manifest target jurisdiction.")
        allowed_components = {"citation", "reasoning", "verdict", "rule_alignment"}
        if set(self.component_weights) != allowed_components:
            raise ValueError("Calibration component weights must declare all four components.")
        if any(value < 0 for value in self.component_weights.values()):
            raise ValueError("Calibration component weights cannot be negative.")
        if sum(self.component_weights.values()) <= 0:
            raise ValueError("Calibration component weights must contain positive mass.")
        if self.evidence_level == "automatic_proxy_pilot" and self.formal_legal_claims_allowed:
            raise ValueError("Automatic proxy calibration cannot release formal legal claims.")
        if self.evidence_level == "expert_adjudicated":
            if self.annotation_summary.get("adjudicated_items", 0) <= 0:
                raise ValueError("Expert calibration requires adjudicated annotation items.")
        if require_expert and self.evidence_level != "expert_adjudicated":
            raise ValueError("This run requires an expert-adjudicated calibration manifest.")

    def client(self, client_id: str) -> ClientRoutingMetadata:
        try:
            return self.client_metadata[client_id]
        except KeyError as exc:
            raise ValueError(f"Calibration metadata missing for client {client_id!r}.") from exc

    def calibrate_verdict_error(self, error_rate: float) -> float:
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError("Probe verdict error must lie in [0, 1].")
        return error_rate

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for client_id, metadata in self.client_metadata.items():
            payload["client_metadata"][client_id]["routing_state"] = metadata.routing_state.value
        payload["limitations"] = list(self.limitations)
        return payload

    def write(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_calibration_manifest(
    path: Path,
    *,
    require_expert: bool = False,
) -> ConflictCalibrationManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    client_metadata = {
        client_id: ClientRoutingMetadata(
            client_id=str(values["client_id"]),
            target_jurisdiction=str(values["target_jurisdiction"]),
            authority_compatibility=float(values["authority_compatibility"]),
            routing_state=RoutingState(str(values["routing_state"])),
        )
        for client_id, values in payload["client_metadata"].items()
    }
    manifest = ConflictCalibrationManifest(
        schema_version=str(payload["schema_version"]),
        calibration_id=str(payload["calibration_id"]),
        evidence_level=str(payload["evidence_level"]),
        formal_legal_claims_allowed=bool(payload["formal_legal_claims_allowed"]),
        task=str(payload["task"]),
        target_jurisdiction=str(payload["target_jurisdiction"]),
        probe_path=str(payload["probe_path"]),
        probe_records=int(payload["probe_records"]),
        exposure_component=str(payload["exposure_component"]),
        exposure_transform=str(payload["exposure_transform"]),
        component_weights={
            str(name): float(value) for name, value in payload["component_weights"].items()
        },
        client_metadata=client_metadata,
        annotation_summary={
            str(name): int(value) for name, value in payload["annotation_summary"].items()
        },
        limitations=tuple(str(value) for value in payload["limitations"]),
    )
    manifest.validate(require_expert=require_expert)
    probe_path = Path(manifest.probe_path)
    if not probe_path.is_absolute():
        probe_path = path.parent / probe_path
    if not probe_path.is_file():
        raise ValueError(f"Calibration probe does not exist: {probe_path}")
    record_count = sum(
        1 for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if record_count != manifest.probe_records:
        raise ValueError("Calibration probe record count does not match the metadata.")
    return manifest


def build_casehold_proxy_manifest(
    *,
    run_dir: Path,
    num_clients: int,
    probe_path: Path,
    target_jurisdiction: str = "us_casehold_common_task",
) -> ConflictCalibrationManifest:
    if not probe_path.is_file():
        raise ValueError(f"CaseHOLD probe does not exist: {probe_path}")
    probe_records = sum(
        1 for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    manifest = ConflictCalibrationManifest(
        schema_version="1.0",
        calibration_id=f"{run_dir.name}-casehold-proxy-v1",
        evidence_level="automatic_proxy_pilot",
        formal_legal_claims_allowed=False,
        task="lex_glue/case_hold",
        target_jurisdiction=target_jurisdiction,
        probe_path=probe_path.name,
        probe_records=probe_records,
        exposure_component="unsupported_pairwise_verdict",
        exposure_transform="gold_error_gated_disagreement",
        component_weights={
            "citation": 0.0,
            "reasoning": 0.0,
            "verdict": 1.0,
            "rule_alignment": 0.0,
        },
        client_metadata={
            f"client_{client_id}": ClientRoutingMetadata(
                client_id=f"client_{client_id}",
                target_jurisdiction=target_jurisdiction,
                authority_compatibility=1.0,
                routing_state=RoutingState.TRANSFERABLE,
            )
            for client_id in range(num_clients)
        },
        annotation_summary={"double_annotated_items": 0, "adjudicated_items": 0},
        limitations=(
            "Gold-error-gated pairwise verdict disagreement is an optimization proxy, not expert legal conflict.",
            "This manifest cannot support cross-jurisdiction or legal-construct validity claims.",
            "All CaseHOLD clients share one task and authority scope in this pilot.",
        ),
    )
    manifest.validate()
    return manifest
