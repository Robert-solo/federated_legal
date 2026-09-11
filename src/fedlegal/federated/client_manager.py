"""Client registration and metadata management."""

from __future__ import annotations

from dataclasses import dataclass

from fedlegal.config.schemas import DataConfig, JurisdictionConfig


@dataclass(frozen=True)
class ClientDescriptor:
    """Federated legal client metadata."""

    client_id: str
    jurisdiction: str
    legal_tradition: str
    institution_type: str
    dataset_names: tuple[str, ...]
    partition_path: str | None = None


class LegalClientManager:
    """Registry for simulated legal institutions."""

    def __init__(self, clients: list[ClientDescriptor]) -> None:
        self._clients = {client.client_id: client for client in clients}

    @classmethod
    def from_data_config(cls, config: DataConfig) -> "LegalClientManager":
        """Create client descriptors from jurisdiction config."""

        return cls([_descriptor_from_jurisdiction(item, config) for item in config.jurisdictions])

    def all(self) -> list[ClientDescriptor]:
        """Return all registered clients in deterministic order."""

        return [self._clients[key] for key in sorted(self._clients)]

    def get(self, client_id: str) -> ClientDescriptor:
        """Return one client descriptor."""

        return self._clients[client_id]

    def ids(self) -> list[str]:
        """Return client IDs in deterministic order."""

        return sorted(self._clients)


def _descriptor_from_jurisdiction(
    item: JurisdictionConfig,
    config: DataConfig,
) -> ClientDescriptor:
    partition_path = config.partition_dir / config.partition_strategy / f"{item.client_id}.jsonl"
    return ClientDescriptor(
        client_id=item.client_id,
        jurisdiction=item.jurisdiction,
        legal_tradition=item.legal_tradition,
        institution_type=item.institution_type,
        dataset_names=tuple(item.dataset_names),
        partition_path=str(partition_path),
    )
