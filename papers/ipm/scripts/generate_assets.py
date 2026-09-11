"""Generate IPM paper tables and figures from current experiment outputs.

The script is intentionally conservative: it uses real output artifacts when
available and writes explicit placeholder/source labels where experiments are
still dry-run or incomplete.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parents[1]
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def copy_existing_figures() -> None:
    src_dir = ROOT / "outputs" / "figures" / "paper"
    for name in [
        "fig1_framework",
        "fig2_conflict_aggregation",
        "fig3_multi_agent_workflow",
        "fig4_experiment_design",
        "fig5_placeholder_results",
    ]:
        src = src_dir / f"{name}.pdf"
        if src.exists():
            shutil.copy2(src, FIG_DIR / f"{name}.pdf")

    eval_svg = ROOT / "outputs" / "evaluation" / "evaluation_default" / "metrics_bar.svg"
    if eval_svg.exists():
        shutil.copy2(eval_svg, FIG_DIR / "fig6_evaluation_metrics.svg")

    conflict_dir = ROOT / "outputs" / "figures" / "conflict_aware_fedavg"
    for i in range(1, 4):
        src = conflict_dir / f"conflict_round_{i:04d}.svg"
        if src.exists():
            shutil.copy2(src, FIG_DIR / f"fig7_conflict_round_{i}.svg")

    five_seed_dir = ROOT / "outputs" / "experiment_analysis" / "fedavg_5seed" / "results"
    five_seed_target = FIG_DIR / "experiments_acl_round2"
    five_seed_target.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in {
        "fedavg_5seed_convergence.pdf": "fig_casehold_fedavg_5seed_convergence.pdf",
        "fedavg_5seed_robustness.pdf": "fig_casehold_fedavg_5seed_robustness.pdf",
        "fedavg_5seed_efficiency.pdf": "fig_casehold_fedavg_5seed_efficiency.pdf",
    }.items():
        src = five_seed_dir / source_name
        if src.exists():
            shutil.copy2(src, five_seed_target / target_name)


def latex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def table_environment(caption: str, label: str, header: list[str], rows: list[list[object]], align: str) -> str:
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""]
    return "\n".join(lines)


def write_dataset_table() -> None:
    rows = [
        ["CaseHOLD", "Same-task heterogeneous FL", "Five-way holding selection", "Not authentic cross-jurisdiction"],
        ["ECtHR-A/B", "Country holdout", "Common ECHR article labels", "Cross-country, shared authority"],
        ["MultiEURLEX", "Language holdout", "Shared multilabel ontology", "Cross-language only"],
        ["LegalBench / LeCaRDv2", "Conflict and citation probes", "Rule, citation, retrieval", "Needs audited mappings"],
        ["Expert comparable probes", "Cross-jurisdiction validation", "Shared task and target authority", "Required for strong transfer claim"],
        ["CAIL2018", "Quarantined", "Downloaded but licence unresolved", "No training claim"],
    ]
    tex = table_environment(
        "Dataset roles and the empirical claims each source can support. Different task label spaces are never averaged as cross-jurisdiction accuracy.",
        "tab:datasets",
        ["Dataset", "Experiment role", "Task / output space", "Claim boundary"],
        rows,
        "llll",
    )
    (TABLE_DIR / "datasets_table.tex").write_text(tex, encoding="utf-8")


def write_metric_table() -> None:
    result_dir = ROOT / "outputs" / "experiment_analysis" / "fedavg_5seed" / "results"
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    if summary:
        accuracy = summary["final_accuracy"]
        rows = [
            ["strategy / model", "FedAvg / Qwen2.5-0.5B", "no FLEN aggregation"],
            ["data / repetitions", "1,200 train; 240 validation; five seeds", "fixed evaluation slice"],
            ["validation accuracy, round 1", "23.92% +/- 0.48", "run-level SD"],
            ["validation accuracy, round 3", f"{100 * accuracy['mean']:.2f}% +/- {100 * accuracy['sd']:.2f}", "95% t interval: 27.95--30.38%"],
            ["descriptive change", f"{100 * summary['round1_to_final_accuracy_gain']:.2f} pp", "no baseline or round-0 attribution"],
            ["evaluation loss reduction", f"{100 * summary['round1_to_final_eval_loss_reduction_fraction']:.1f}%", "round 1 to round 3"],
            ["mean update norm", f"{summary['mean_client_drift_l2']['mean']:.4f} +/- {summary['mean_client_drift_l2']['sd']:.4f}", "L2 from broadcast adapter"],
            ["logical tensor payload", f"{summary['communication_mib']['mean']:.2f} MiB per seed", "excludes protocol overhead"],
            ["privacy / agents", "not enabled", "data locality only"],
        ]
    else:
        rows = [["five-seed CaseHOLD analysis", "N/A", "run scripts/analyze_fedavg_5seed.py"]]
    tex = table_environment(
        "Measured outputs from five bounded CaseHOLD Flower/PEFT FedAvg runs. Uncertainty is run-level standard deviation across seeds 41--45.",
        "tab:current-evaluation",
        ["Metric", "Value", "Source detail"],
        rows,
        "lll",
    )
    (TABLE_DIR / "current_evaluation_table.tex").write_text(tex, encoding="utf-8")


def write_conflict_table() -> None:
    log_path = ROOT / "outputs" / "logs" / "aggregation" / "conflict_aware_fedavg_aggregation.jsonl"
    rows: list[list[object]] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            report = event["conflict_report"]
            rows.append(
                [
                    event["round_id"],
                    f"{report['citation_conflict']:.4f}",
                    f"{report['reasoning_conflict']:.4f}",
                    f"{report['verdict_conflict']:.4f}",
                    f"{report['rule_alignment_distance']:.4f}",
                    f"{report['total_conflict']:.4f}",
                ]
            )
    tex = table_environment(
        "Conflict-aware aggregation diagnostics aligned with the current conflict-aware FedAvg output log.",
        "tab:conflict-diagnostics",
        ["Round", "Citation", "Reasoning", "Verdict", "Rule align.", "Total"],
        rows or [["N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]],
        "lccccc",
    )
    (TABLE_DIR / "conflict_diagnostics_table.tex").write_text(tex, encoding="utf-8")


def write_baseline_and_ablation_tables() -> None:
    baseline_rows = [
        ["Centralized legal encoder / LLM", "yes", "yes", "yes", "yes", "N/A", "N/A"],
        ["FedAvg / FedProx / SCAFFOLD / FedNova", "yes", "yes", "yes", "yes", "yes", "yes"],
        ["LoRA / adapter / prompt tuning", "yes", "yes", "yes", "yes", "yes", "N/A"],
        ["Single-agent legal reasoning", "yes", "yes", "yes", "yes", "N/A", "N/A"],
        ["Full federated multi-agent framework", "yes", "yes", "yes", "yes", "yes", "yes"],
    ]
    (TABLE_DIR / "baseline_metric_matrix.tex").write_text(
        table_environment(
            "Baseline and metric matrix for empirical reporting.",
            "tab:baseline-metric-matrix",
            ["Baseline family", "Accuracy", "Citation", "Coherence", "Hallucination", "Comm. cost", "Drift"],
            baseline_rows,
            "lllllll",
        ),
        encoding="utf-8",
    )
    ablation_rows = [
        ["FedAvg + LoRA", "TBD", "TBD", "TBD", "TBD"],
        ["Conflict-aware aggregation", "TBD", "TBD", "TBD", "TBD"],
        ["Full framework", "TBD", "TBD", "TBD", "TBD"],
        ["Full without citation verifier", "TBD", "TBD", "TBD", "TBD"],
        ["Full without jurisdiction embedding", "TBD", "TBD", "TBD", "TBD"],
    ]
    (TABLE_DIR / "ablation_table.tex").write_text(
        table_environment(
            "Ablation reporting template. Values should be filled after completed runs under the same data split and random-seed protocol.",
            "tab:ablations",
            ["Variant", "Accuracy $\\uparrow$", "Citation consistency $\\uparrow$", "Hallucination $\\downarrow$", "Communication $\\downarrow$"],
            ablation_rows,
            "lllll",
        ),
        encoding="utf-8",
    )
    config_rows = [
        ["Multi-dataset figure CSVs", "Synthetic templates", "Fixed-seed layout generator", "None"],
        ["Conflict-aware Flower strategy", "Not implemented", "Runtime falls back to FedAvg", "None"],
        ["Multi-agent workflow", "Prototype / heuristic", "Typed state and logs", "Interface feasibility"],
        ["CaseHOLD PEFT runs", "Measured, five seeds", "FedAvg, four clients, three rounds", "Executability and seed reproducibility"],
        ["DP and secure aggregation", "Configuration only", "No accountant or protocol runtime", "None"],
    ]
    (TABLE_DIR / "experiment_config_table.tex").write_text(
        table_environment(
            "Artifact status at the time of revision. A manifest, configuration flag, or dry run is not counted as empirical validation.",
            "tab:experiment-configs",
            ["Artifact", "Status", "Implemented mechanism", "Permitted claim"],
            config_rows,
            "llll",
        ),
        encoding="utf-8",
    )


def plot_conflict_trends() -> None:
    log_path = ROOT / "outputs" / "logs" / "aggregation" / "conflict_aware_fedavg_aggregation.jsonl"
    rounds: list[int] = []
    totals: list[float] = []
    reasoning: list[float] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            report = event["conflict_report"]
            rounds.append(event["round_id"])
            totals.append(report["total_conflict"])
            reasoning.append(report["reasoning_conflict"])
    if not rounds:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(rounds, totals, marker="o", label="total conflict", color="#C9805C")
    ax.plot(rounds, reasoning, marker="o", label="reasoning conflict", color="#2F6F9F")
    ax.set_xlabel("federated round")
    ax.set_ylabel("conflict score")
    ax.set_title("Conflict-aware aggregation diagnostics")
    ax.grid(axis="y", color="#E8EDF3", lw=0.7)
    ax.legend()
    fig.savefig(FIG_DIR / "fig6_conflict_trends.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_conflict_trends.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    copy_existing_figures()
    write_dataset_table()
    write_metric_table()
    write_conflict_table()
    write_baseline_and_ablation_tables()
    plot_conflict_trends()
    print(f"Wrote IPM paper assets to {PAPER_DIR}")


if __name__ == "__main__":
    main()
