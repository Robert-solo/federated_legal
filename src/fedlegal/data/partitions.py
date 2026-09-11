"""Federated non-IID partitioning for processed legal datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fedlegal.config.schemas import DataConfig
from fedlegal.data.io import read_jsonl, write_json, write_jsonl


@dataclass(frozen=True)
class PartitionPlan:
    """Serializable description of simulated client partitions."""

    strategy: str
    alpha: float
    client_ids: tuple[str, ...]
    jurisdictions: tuple[str, ...]
    output_dir: str | None = None
    client_sizes: dict[str, int] | None = None


def build_partition_plan(config: DataConfig) -> PartitionPlan:
    """Build a non-IID partition plan from dataset configuration."""

    return PartitionPlan(
        strategy=config.partition_strategy,
        alpha=config.dirichlet_alpha,
        client_ids=tuple(item.client_id for item in config.jurisdictions),
        jurisdictions=tuple(item.jurisdiction for item in config.jurisdictions),
    )


def partition_processed_datasets(config: DataConfig) -> PartitionPlan:
    """Partition processed JSONL datasets into federated client files."""

    records = _load_processed_records(config.processed_dir)
    client_ids = [item.client_id for item in config.jurisdictions]
    if not client_ids:
        raise ValueError("At least one jurisdiction client is required for partitioning.")

    if config.partition_strategy in {"jurisdiction_holdout", "jurisdiction_based"}:
        assignments = _jurisdiction_based_partition(records, config)
    elif config.partition_strategy == "institution_based":
        assignments = _institution_based_partition(records, config)
    elif config.partition_strategy in {"dirichlet", "label_skew"}:
        assignments = _dirichlet_partition(
            records=records,
            client_ids=client_ids,
            alpha=config.dirichlet_alpha,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unsupported partition strategy: {config.partition_strategy}")

    output_dir = config.partition_dir / config.partition_strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    client_sizes: dict[str, int] = {}
    for client_id in client_ids:
        client_records = assignments.get(client_id, [])
        client_sizes[client_id] = len(client_records)
        write_jsonl(output_dir / f"{client_id}.jsonl", client_records)

    manifest = {
        "strategy": config.partition_strategy,
        "seed": config.seed,
        "dirichlet_alpha": config.dirichlet_alpha,
        "client_sizes": client_sizes,
    }
    write_json(output_dir / "manifest.json", manifest)
    return PartitionPlan(
        strategy=config.partition_strategy,
        alpha=config.dirichlet_alpha,
        client_ids=tuple(client_ids),
        jurisdictions=tuple(item.jurisdiction for item in config.jurisdictions),
        output_dir=str(output_dir),
        client_sizes=client_sizes,
    )


def _load_processed_records(processed_dir: Path) -> list[dict]:
    paths = sorted(path for path in processed_dir.rglob("*.jsonl") if path.is_file())
    records: list[dict] = []
    for path in paths:
        records.extend(read_jsonl(path))
    if not records:
        raise FileNotFoundError(f"No processed JSONL records found under {processed_dir}")
    return records


def _jurisdiction_based_partition(records: list[dict], config: DataConfig) -> dict[str, list[dict]]:
    by_jurisdiction = {item.jurisdiction: item.client_id for item in config.jurisdictions}
    fallback = config.jurisdictions[0].client_id
    assignments: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        client_id = by_jurisdiction.get(str(record.get("jurisdiction")), fallback)
        assignments[client_id].append(record)
    return dict(assignments)


def _institution_based_partition(records: list[dict], config: DataConfig) -> dict[str, list[dict]]:
    known_clients = {item.client_id for item in config.jurisdictions}
    fallback = config.jurisdictions[0].client_id
    assignments: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        institution_id = str(record.get("institution_id") or fallback)
        client_id = institution_id if institution_id in known_clients else fallback
        assignments[client_id].append(record)
    return dict(assignments)


def _dirichlet_partition(
    records: list[dict],
    client_ids: list[str],
    alpha: float,
    seed: int,
) -> dict[str, list[dict]]:
    if alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive.")

    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        label = record.get("label")
        if isinstance(label, list):
            label_key = "|".join(str(item) for item in label)
        elif label is None:
            label_key = str(record.get("dataset", "unlabeled"))
        else:
            label_key = str(label)
        grouped[label_key].append(record)

    assignments: dict[str, list[dict]] = {client_id: [] for client_id in client_ids}
    for label_records in grouped.values():
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        proportions = _sample_dirichlet(len(client_ids), alpha, rng)
        counts = _proportions_to_counts(len(shuffled), proportions)

        start = 0
        for client_id, count in zip(client_ids, counts, strict=True):
            end = start + count
            assignments[client_id].extend(shuffled[start:end])
            start = end

    for client_records in assignments.values():
        rng.shuffle(client_records)
    return assignments


def _sample_dirichlet(size: int, alpha: float, rng: random.Random) -> list[float]:
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(samples)
    if total == 0:
        return [1.0 / size] * size
    return [sample / total for sample in samples]


def _proportions_to_counts(total: int, proportions: list[float]) -> list[int]:
    raw_counts = [int(total * proportion) for proportion in proportions]
    remainder = total - sum(raw_counts)
    ranked = sorted(
        range(len(proportions)),
        key=lambda index: (total * proportions[index]) - raw_counts[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        raw_counts[index] += 1
    return raw_counts
