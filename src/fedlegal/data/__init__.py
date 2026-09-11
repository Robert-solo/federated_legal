"""Judicial data and non-IID jurisdiction partitioning."""

from fedlegal.data.citations import extract_citations
from fedlegal.data.partitions import (
    PartitionPlan,
    build_partition_plan,
    partition_processed_datasets,
)
from fedlegal.data.preprocessing import preprocess_datasets
from fedlegal.data.records import DatasetManifest, LegalDatasetRecord

__all__ = [
    "DatasetManifest",
    "LegalDatasetRecord",
    "PartitionPlan",
    "build_partition_plan",
    "extract_citations",
    "partition_processed_datasets",
    "preprocess_datasets",
]
