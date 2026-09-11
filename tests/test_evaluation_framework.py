import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fedlegal.config import load_experiment_config
from fedlegal.evaluation import (
    EvaluationRunner,
    citation_accuracy,
    client_drift,
    communication_cost,
    hallucination_rate,
    legal_consistency,
    load_examples,
)


def test_evaluation_metrics_compute_expected_values() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        predictions = root / "predictions.jsonl"
        communication = root / "communication.jsonl"
        _write_jsonl(
            predictions,
            [
                {
                    "example_id": "case-1",
                    "predicted_label": "grant",
                    "gold_label": "grant",
                    "predicted_citations": ["42 U.S.C. § 1983"],
                    "gold_citations": ["42 U.S.C. § 1983"],
                    "generated_text": "The claim is supported by 42 U.S.C. § 1983.",
                    "supported_facts": ["claim is supported"],
                },
                {
                    "example_id": "case-2",
                    "predicted_label": "deny",
                    "gold_label": "grant",
                    "predicted_citations": ["Article 6"],
                    "gold_citations": ["Clause 12"],
                    "generated_text": "Unsupported invented rationale.",
                    "supported_facts": ["contract assignment"],
                },
            ],
        )
        _write_jsonl(
            communication,
            [
                {"round_id": 1, "client_id": "a", "num_bytes": 10, "metrics": {"update_norm": 1.0}},
                {"round_id": 1, "client_id": "b", "num_bytes": 30, "metrics": {"update_norm": 3.0}},
            ],
        )

        examples = load_examples(predictions)

        assert citation_accuracy(examples).value == 0.5
        assert legal_consistency(examples).value == 0.5
        assert hallucination_rate(examples).value == 0.5
        assert communication_cost(communication).value == 40.0
        assert client_drift(communication).value == 2.0


def test_evaluation_runner_writes_report_artifacts() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        predictions = root / "predictions.jsonl"
        communication = root / "communication.jsonl"
        _write_jsonl(
            predictions,
            [
                {
                    "example_id": "case-1",
                    "predicted_label": "grant",
                    "gold_label": "grant",
                    "predicted_citations": ["42 U.S.C. § 1983"],
                    "gold_citations": ["42 U.S.C. § 1983"],
                    "generated_text": "The claim is supported by 42 U.S.C. § 1983.",
                    "supported_facts": ["claim is supported"],
                }
            ],
        )
        _write_jsonl(
            communication,
            [{"round_id": 1, "client_id": "a", "num_bytes": 12, "metrics": {"client_drift": 0.2}}],
        )
        config = load_experiment_config("configs/experiments/evaluation_default.yaml")
        config.evaluation.output_dir = root / "evaluation"

        report = EvaluationRunner(config).run(predictions, communication)
        output_dir = Path(report.output_dir)

        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "metrics_table.csv").exists()
        assert (output_dir / "metrics_table.md").exists()
        assert (output_dir / "metrics_bar.svg").exists()
        assert (output_dir / "report.md").exists()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
