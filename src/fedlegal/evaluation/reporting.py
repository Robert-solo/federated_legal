"""Automatic evaluation report generation."""

from __future__ import annotations

import json
from pathlib import Path

from fedlegal.config.schemas import ExperimentConfig
from fedlegal.evaluation.metrics import (
    aggregation_stability,
    citation_accuracy,
    citation_consistency,
    client_drift,
    communication_cost,
    cross_jurisdiction_generalization,
    exact_match_f1,
    hallucination_rate,
    legal_accuracy,
    legal_consistency,
    micro_f1,
    load_examples,
    privacy_leakage_risk,
    reasoning_coherence,
)
from fedlegal.evaluation.plots import write_metric_bar_plot
from fedlegal.evaluation.schemas import EvaluationReport, MetricResult
from fedlegal.evaluation.tables import write_metric_tables


class EvaluationRunner:
    """Compute metrics, plots, tables, and report artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.evaluation.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        predictions_path: str | Path | None = None,
        communication_log_path: str | Path | None = None,
    ) -> EvaluationReport:
        """Run the evaluation framework."""

        examples = load_examples(predictions_path or self.config.evaluation.predictions_path)
        communication_log = (
            communication_log_path
            or self.config.evaluation.communication_log_path
            or self.config.federated.communication_log_dir / f"{self.config.name}.jsonl"
        )
        metrics = {
            "legal_accuracy": legal_accuracy(examples),
            "citation_accuracy": citation_accuracy(examples),
            "citation_consistency": citation_consistency(examples),
            "legal_consistency": legal_consistency(examples),
            "micro_f1": micro_f1(examples),
            "exact_match_f1": exact_match_f1(examples),
            "reasoning_coherence": reasoning_coherence(examples),
            "hallucination_rate": hallucination_rate(examples),
            "communication_cost": communication_cost(communication_log),
            "client_drift": client_drift(communication_log),
            "cross_jurisdiction_generalization": cross_jurisdiction_generalization(examples),
            "privacy_leakage_risk": privacy_leakage_risk(examples),
            "aggregation_stability": aggregation_stability(
                self.config.evaluation.aggregation_log_path or communication_log
            ),
        }
        report = EvaluationReport(
            run_name=self.config.name,
            metrics=metrics,
            num_examples=len(examples),
            output_dir=str(self.output_dir),
        )
        self._write_artifacts(report)
        return report

    def _write_artifacts(self, report: EvaluationReport) -> None:
        metric_json = {
            name: metric.model_dump(mode="json") for name, metric in report.metrics.items()
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metric_json, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        table_paths = write_metric_tables(report.metrics, self.output_dir)
        plot_path = write_metric_bar_plot(report.metrics, self.output_dir)
        self._write_markdown_report(report, table_paths, plot_path)

    def _write_markdown_report(
        self,
        report: EvaluationReport,
        table_paths: dict[str, str],
        plot_path: str,
    ) -> None:
        lines = [
            f"# Evaluation Report: {report.run_name}",
            "",
            f"Examples evaluated: {report.num_examples}",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        for metric in report.metrics.values():
            lines.append(f"| {metric.name} | {metric.value:.6f} |")
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                f"- Metrics JSON: `{self.output_dir / 'metrics.json'}`",
                f"- CSV table: `{table_paths['csv']}`",
                f"- Markdown table: `{table_paths['markdown']}`",
                f"- SVG plot: `{plot_path}`",
            ]
        )
        (self.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
