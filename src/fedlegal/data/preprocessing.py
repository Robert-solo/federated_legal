"""Legal dataset preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

from fedlegal.config.schemas import DataConfig
from fedlegal.data.adapters import DatasetAdapter
from fedlegal.data.io import write_json, write_jsonl
from fedlegal.data.records import DatasetManifest, LegalDatasetRecord
from fedlegal.data.tokenization import LegalTokenizer


def preprocess_datasets(
    config: DataConfig,
    limit_per_source: int | None = None,
) -> list[DatasetManifest]:
    """Preprocess configured legal datasets into `data/processed`.

    The pipeline performs dataset normalization, jurisdiction tagging, citation
    extraction, tokenizer encoding, and JSONL persistence.
    """

    tokenizer = LegalTokenizer(config.tokenizer)
    manifests: list[DatasetManifest] = []
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    for source in config.datasets:
        records = DatasetAdapter(source, config.jurisdictions, config.raw_dir).iter_load(
            limit=limit_per_source
        )
        processed = [_tokenize_record(record, tokenizer) for record in records]
        output_path = _processed_path(config.processed_dir, source.name, source.split, source.task)
        write_jsonl(output_path, (record.model_dump(mode="json") for record in processed))

        manifest = DatasetManifest(
            name=source.name,
            task=source.task,
            split=source.split,
            output_path=str(output_path),
            num_records=len(processed),
            jurisdictions=sorted({record.jurisdiction for record in processed}),
            institutions=sorted({record.institution_id for record in processed}),
        )
        manifests.append(manifest)

    write_json(
        config.processed_dir / "manifest.json",
        {"datasets": [manifest.model_dump(mode="json") for manifest in manifests]},
    )
    return manifests


def _tokenize_record(record: LegalDatasetRecord, tokenizer: LegalTokenizer) -> LegalDatasetRecord:
    encoded = tokenizer.encode(record.text)
    return record.model_copy(
        update={
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "tokens": encoded.tokens,
        }
    )


def _processed_path(processed_dir: Path, dataset_name: str, split: str, task: str | None) -> Path:
    if task:
        return processed_dir / dataset_name / task / f"{split}.jsonl"
    return processed_dir / dataset_name / f"{split}.jsonl"
