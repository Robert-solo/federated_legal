import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fedlegal.config.schemas import (
    DataConfig,
    DatasetSourceConfig,
    JurisdictionConfig,
    TokenizerConfig,
)
from fedlegal.data import partition_processed_datasets, preprocess_datasets


def test_preprocess_and_partition_supported_datasets() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw_dir = root / "raw"
        processed_dir = root / "processed"
        partition_dir = root / "partitions"

        _write_jsonl(
            raw_dir / "CAIL" / "train.jsonl",
            [{"id": "cn-1", "fact": "依据《民法典》第五百条处理。", "accusation": "contract"}],
        )
        _write_jsonl(
            raw_dir / "LexGLUE" / "train.jsonl",
            [{"id": "eu-1", "text": "Article 6 applies.", "label": "privacy"}],
        )
        _write_jsonl(
            raw_dir / "CaseHOLD" / "train.jsonl",
            [{"id": "us-1", "prompt": "See 42 U.S.C. § 1983.", "label": "civil_rights"}],
        )
        _write_jsonl(
            raw_dir / "CUAD" / "train.jsonl",
            [{"id": "cuad-1", "clause": "Clause 12 controls assignment.", "clause_type": "assignment"}],
        )
        _write_jsonl(
            raw_dir / "LeCaRDv2" / "train.jsonl",
            [{"id": "lecard-1", "text": "刑事判决书", "label": 1}],
        )
        _write_jsonl(
            raw_dir / "LegalBench" / "train.jsonl",
            [{"id": "legalbench-1", "text": "Is precedent controlling?", "answer": "Yes"}],
        )
        _write_jsonl(
            raw_dir / "MultiEURLEX" / "train.jsonl",
            [{"id": "eurlex-1", "text": "EU regulation", "parallel_text": "EU-Verordnung", "label": [1]}],
        )

        config = DataConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            partition_dir=partition_dir,
            partition_strategy="institution_based",
            tokenizer=TokenizerConfig(backend="simple", max_length=128),
            datasets=[
                DatasetSourceConfig(
                    name="CAIL",
                    raw_path=raw_dir / "CAIL" / "train.jsonl",
                    text_fields=["fact"],
                    label_field="accusation",
                    citation_fields=["articles"],
                    jurisdiction="chinese_civil_law",
                    institution_id="court_cn_01",
                    institution_type="court",
                ),
                DatasetSourceConfig(
                    name="LexGLUE",
                    raw_path=raw_dir / "LexGLUE" / "train.jsonl",
                    text_fields=["text"],
                    label_field="label",
                    jurisdiction="european_regulatory_law",
                    institution_id="institute_eu_01",
                    institution_type="legal_research_institute",
                ),
                DatasetSourceConfig(
                    name="CaseHOLD",
                    raw_path=raw_dir / "CaseHOLD" / "train.jsonl",
                    text_fields=["prompt"],
                    label_field="label",
                    jurisdiction="us_common_law",
                    institution_id="court_us_01",
                    institution_type="court",
                ),
                DatasetSourceConfig(
                    name="CUAD",
                    raw_path=raw_dir / "CUAD" / "train.jsonl",
                    text_fields=["clause"],
                    label_field="clause_type",
                    jurisdiction="contract_law",
                    institution_id="firm_contract_01",
                    institution_type="law_firm",
                ),
                DatasetSourceConfig(
                    name="LeCaRDv2",
                    raw_path=raw_dir / "LeCaRDv2" / "train.jsonl",
                    jurisdiction="chinese_civil_law",
                    institution_id="court_cn_01",
                    institution_type="court",
                ),
                DatasetSourceConfig(
                    name="LegalBench",
                    raw_path=raw_dir / "LegalBench" / "train.jsonl",
                    jurisdiction="us_common_law",
                    institution_id="court_us_01",
                    institution_type="court",
                ),
                DatasetSourceConfig(
                    name="MultiEURLEX",
                    raw_path=raw_dir / "MultiEURLEX" / "train.jsonl",
                    jurisdiction="european_regulatory_law",
                    institution_id="institute_eu_01",
                    institution_type="legal_research_institute",
                ),
            ],
            jurisdictions=[
                JurisdictionConfig(
                    client_id="court_cn_01",
                    jurisdiction="chinese_civil_law",
                    legal_tradition="civil_law",
                    institution_type="court",
                    dataset_names=["CAIL"],
                ),
                JurisdictionConfig(
                    client_id="institute_eu_01",
                    jurisdiction="european_regulatory_law",
                    legal_tradition="civil_law_regulatory",
                    institution_type="legal_research_institute",
                    dataset_names=["LexGLUE"],
                ),
                JurisdictionConfig(
                    client_id="court_us_01",
                    jurisdiction="us_common_law",
                    legal_tradition="common_law",
                    institution_type="court",
                    dataset_names=["CaseHOLD"],
                ),
                JurisdictionConfig(
                    client_id="firm_contract_01",
                    jurisdiction="contract_law",
                    legal_tradition="mixed_commercial",
                    institution_type="law_firm",
                    dataset_names=["CUAD"],
                ),
            ],
        )

        manifests = preprocess_datasets(config)
        plan = partition_processed_datasets(config)

        assert sum(manifest.num_records for manifest in manifests) == 7
        assert (processed_dir / "CAIL" / "train.jsonl").exists()
        assert plan.client_sizes == {
            "court_cn_01": 2,
            "institute_eu_01": 2,
            "court_us_01": 2,
            "firm_contract_01": 1,
        }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
