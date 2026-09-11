"""Analyze matched 20-round CaseHOLD federated screening runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


MIB = 1024**2
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
METHODS = {
    "FedAvg": "E1_casehold_fedavg_20r_20260901_seed{seed}",
    "FedProx": "E1_casehold_fedprox_20r_20260901_seed{seed}",
    "Target cohort ($\\lambda=0$)": (
        "E1_casehold_target_cohort_20r_20260901_v2_seed{seed}"
    ),
    "Conflict only": "E1_casehold_conflict_only_20r_20260901_v1_seed{seed}",
    "Personalization only": (
        "E1_casehold_personalization_only_20r_20260901_v1_seed{seed}"
    ),
    "FLEN": "E1_casehold_flen_20r_20260901_v2_seed{seed}",
    "Personalization guarded": (
        "E1_casehold_personalization_guarded_20r_20260901_v1_seed{seed}"
    ),
    "FLEN guarded": "E1_casehold_flen_guarded_20r_20260901_v1_seed{seed}",
}
COLORS = {
    "FedAvg": "#5B7FA3",
    "FedProx": "#8D9AA6",
    "Target cohort ($\\lambda=0$)": "#C4865A",
    "Conflict only": "#A8663D",
    "Personalization only": "#78A88B",
    "FLEN": "#4E9470",
    "Personalization guarded": "#6E9FB4",
    "FLEN guarded": "#2F7657",
}
CONTRASTS = {
    "conflict_10_shared": ("Conflict only", "Target cohort ($\\lambda=0$)"),
    "conflict_8_plus_2": ("FLEN", "Personalization only"),
    "personalization_lambda_zero": (
        "Personalization only",
        "Target cohort ($\\lambda=0$)",
    ),
    "full_flen": ("FLEN", "Target cohort ($\\lambda=0$)"),
    "guarded_conflict": ("FLEN guarded", "Personalization guarded"),
    "guarded_personalization": (
        "Personalization guarded",
        "Target cohort ($\\lambda=0$)",
    ),
    "guarded_full_flen": ("FLEN guarded", "Target cohort ($\\lambda=0$)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/real_flower_peft")
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment_analysis/e1_flen_screening",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd_ci(values: Iterable[float]) -> tuple[float, float, float]:
    data = [float(value) for value in values]
    mean = statistics.fmean(data)
    if len(data) < 2:
        return mean, 0.0, 0.0
    standard_deviation = statistics.stdev(data)
    critical = T_CRITICAL_95.get(len(data), 1.96)
    return mean, standard_deviation, critical * standard_deviation / math.sqrt(len(data))


def collect_runs(
    input_root: Path,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    for method, template in METHODS.items():
        for seed in seeds:
            run_dir = input_root / template.format(seed=seed)
            acceptance = read_json(run_dir / "acceptance.json")
            if acceptance.get("status") != "passed":
                raise ValueError(f"Run did not pass acceptance: {run_dir}")
            report = read_json(run_dir / "real_experiment_report.json")
            config = read_json(run_dir / "resolved_config.json")
            server_eval = read_jsonl(run_dir / "server_eval_rounds.jsonl")
            client_eval = read_jsonl(run_dir / "client_eval_log.jsonl")
            communication = read_jsonl(run_dir / "communication_log.jsonl")
            aggregation_path = run_dir / "aggregation_rounds.jsonl"
            aggregation = read_jsonl(aggregation_path) if aggregation_path.is_file() else []
            if len(server_eval) != 20 or len(client_eval) != 80 or len(communication) != 80:
                raise ValueError(f"Incomplete 20-round logs: {run_dir}")
            if method not in {"FedAvg", "FedProx"} and len(aggregation) != 20:
                raise ValueError(f"Incomplete aggregation audit: {run_dir}")

            client_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
            comm_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
            aggregation_by_round = {int(row["round"]): row for row in aggregation}
            for row in client_eval:
                client_by_round[int(row["metrics"]["server_round"])].append(row)
            for row in communication:
                comm_by_round[int(row["metrics"]["server_round"])].append(row)

            for server_row in sorted(server_eval, key=lambda row: int(row["round"])):
                round_id = int(server_row["round"])
                client_rows = client_by_round[round_id]
                comm_rows = comm_by_round[round_id]
                client_accuracies = [float(row["metrics"]["accuracy"]) for row in client_rows]
                shared_client_accuracies = [
                    float(row["metrics"]["shared_accuracy"])
                    for row in client_rows
                    if "shared_accuracy" in row["metrics"]
                ]
                personalization_gains = [
                    float(row["metrics"]["personalization_accuracy_gain"])
                    for row in client_rows
                    if "personalization_accuracy_gain" in row["metrics"]
                ]
                audit = aggregation_by_round.get(round_id)
                conflict_mean = None
                conflict_spread = None
                max_weight_delta = None
                probe_pair_coverage = None
                if audit is not None:
                    exposures = [
                        float(value)
                        for value in audit["conflict_exposures"].values()
                        if value is not None
                    ]
                    conflict_mean = statistics.fmean(exposures)
                    conflict_spread = max(exposures) - min(exposures)
                    max_weight_delta = max(
                        abs(
                            float(audit["final_weights"][client_id])
                            - float(audit["base_weights"][client_id])
                        )
                        for client_id in audit["base_weights"]
                    )
                    probe_pair_coverage = float(audit["probe_pair_coverage"])
                round_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "round": round_id,
                        "server_accuracy": float(server_row["metrics"]["accuracy"]),
                        "server_eval_loss": float(server_row["metrics"]["eval_loss"]),
                        "client_accuracy_mean": statistics.fmean(client_accuracies),
                        "client_accuracy_worst": min(client_accuracies),
                        "client_shared_accuracy_mean": (
                            statistics.fmean(shared_client_accuracies)
                            if len(shared_client_accuracies) == len(client_rows)
                            else None
                        ),
                        "personalization_accuracy_gain_mean": (
                            statistics.fmean(personalization_gains)
                            if len(personalization_gains) == len(client_rows)
                            else None
                        ),
                        "mean_client_drift_l2": statistics.fmean(
                            float(row["metrics"]["client_drift_l2"])
                            for row in comm_rows
                        ),
                        "mean_train_loss": statistics.fmean(
                            float(row["metrics"]["train_loss"]) for row in comm_rows
                        ),
                        "round_communication_mib": sum(
                            int(row["num_bytes"]) for row in comm_rows
                        )
                        / MIB,
                        "conflict_exposure_mean": conflict_mean,
                        "conflict_exposure_spread": conflict_spread,
                        "max_weight_delta": max_weight_delta,
                        "probe_pair_coverage": probe_pair_coverage,
                    }
                )

            final = [
                row
                for row in round_rows
                if row["method"] == method and row["seed"] == seed and row["round"] == 20
            ][0]
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "server_final_accuracy": final["server_accuracy"],
                    "client_final_accuracy_mean": final["client_accuracy_mean"],
                    "client_final_accuracy_worst": final["client_accuracy_worst"],
                    "client_shared_final_accuracy_mean": final[
                        "client_shared_accuracy_mean"
                    ],
                    "personalization_final_accuracy_gain_mean": final[
                        "personalization_accuracy_gain_mean"
                    ],
                    "best_server_accuracy": float(report["best_accuracy"]),
                    "communication_mib": int(report["communication_bytes"]) / MIB,
                    "mean_client_drift_l2": float(report["mean_client_drift_l2"]),
                    "local_personalization_steps": int(
                        config.get("local_personalization_steps", 0)
                    ),
                    "formal_legal_claims_allowed": bool(
                        report.get("formal_legal_claims_allowed", False)
                    ),
                    "run_name": report["run_name"],
                }
            )
    return seed_rows, round_rows


def aggregate_methods(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["method"])].append(row)
    summary_rows = []
    metrics = (
        "server_final_accuracy",
        "client_final_accuracy_mean",
        "client_final_accuracy_worst",
        "communication_mib",
        "mean_client_drift_l2",
    )
    for method in METHODS:
        rows = grouped[method]
        result: dict[str, Any] = {"method": method, "seeds": len(rows)}
        for metric in metrics:
            mean, standard_deviation, ci95 = mean_sd_ci(float(row[metric]) for row in rows)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = standard_deviation
            result[f"{metric}_ci95"] = ci95
        summary_rows.append(result)
    return summary_rows


def paired_effects(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["method"]), int(row["seed"])): row for row in seed_rows}
    result: dict[str, Any] = {}
    for contrast_name, (left_method, right_method) in CONTRASTS.items():
        seeds = sorted(
            seed
            for method, seed in by_key
            if method == left_method and (right_method, seed) in by_key
        )
        contrast: dict[str, Any] = {
            "left_method": left_method,
            "right_method": right_method,
            "seeds": seeds,
        }
        for metric in (
            "server_final_accuracy",
            "client_final_accuracy_mean",
            "client_final_accuracy_worst",
        ):
            differences = [
                float(by_key[(left_method, seed)][metric])
                - float(by_key[(right_method, seed)][metric])
                for seed in seeds
            ]
            mean, standard_deviation, ci95 = mean_sd_ci(differences)
            contrast[metric] = {
                "differences": differences,
                "mean": mean,
                "sd": standard_deviation,
                "ci95_half_width": ci95,
                "ci95_excludes_zero": mean - ci95 > 0 or mean + ci95 < 0,
            }
        result[contrast_name] = contrast
    return result


def paired_local_residual_effect(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in (
        "Personalization only",
        "Personalization guarded",
        "FLEN guarded",
    ):
        values = [
            float(row["personalization_final_accuracy_gain_mean"])
            for row in seed_rows
            if row["method"] == method
            and row["personalization_final_accuracy_gain_mean"] is not None
        ]
        mean, standard_deviation, ci95 = mean_sd_ci(values)
        result[method] = {
            "seed_values": values,
            "mean": mean,
            "sd": standard_deviation,
            "ci95_half_width": ci95,
            "ci95_excludes_zero": mean - ci95 > 0 or mean + ci95 < 0,
        }
    return result


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E6EAF0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=400)
    plt.close(fig)


def plot_convergence(round_rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    method_groups = (
        (
            "Shared and aggregation controls",
            ("FedAvg", "FedProx", "Target cohort ($\\lambda=0$)", "Conflict only"),
        ),
        (
            "Personalization ablations",
            ("Personalization only", "FLEN", "Personalization guarded", "FLEN guarded"),
        ),
    )
    metrics = (
        ("server_accuracy", "Aggregated client accuracy"),
        ("client_accuracy_worst", "Worst client accuracy"),
    )
    for column, (group_title, methods) in enumerate(method_groups):
        for method in methods:
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in round_rows:
                if row["method"] == method:
                    grouped[int(row["round"])].append(row)
            rounds = sorted(grouped)
            for row_index, (metric, _) in enumerate(metrics):
                means = [
                    100 * statistics.fmean(float(row[metric]) for row in grouped[round_id])
                    for round_id in rounds
                ]
                standard_deviations = [
                    100 * statistics.stdev(
                        float(row[metric]) for row in grouped[round_id]
                    )
                    for round_id in rounds
                ]
                axis = axes[row_index, column]
                axis.plot(rounds, means, color=COLORS[method], lw=1.8, label=method)
                axis.fill_between(
                    rounds,
                    np.array(means) - np.array(standard_deviations),
                    np.array(means) + np.array(standard_deviations),
                    color=COLORS[method],
                    alpha=0.1,
                )
        axes[0, column].set_title(group_title, loc="left", fontweight="bold")
        axes[0, column].legend(fontsize=6.6, ncol=2)
    for row_index, (_, metric_label) in enumerate(metrics):
        for column in range(2):
            axes[row_index, column].set_ylabel(f"{metric_label} (%) $\\uparrow$")
            axes[row_index, column].set_xticks([1, 5, 10, 15, 20])
            if row_index == 1:
                axes[row_index, column].set_xlabel("Federated round")
    fig.tight_layout()
    save_figure(fig, output_dir, "e1_accuracy_convergence")


def plot_final_summary(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    methods = list(METHODS)
    positions = np.arange(len(methods))
    for axis, metric, title in (
        (axes[0], "server_final_accuracy", "Aggregated final accuracy"),
        (axes[1], "client_final_accuracy_worst", "Worst client accuracy"),
    ):
        means = [100 * float(row[f"{metric}_mean"]) for row in summary_rows]
        ci95 = [100 * float(row[f"{metric}_ci95"]) for row in summary_rows]
        axis.bar(
            positions,
            means,
            yerr=ci95,
            capsize=3,
            color=[COLORS[method] for method in methods],
            alpha=0.9,
        )
        axis.set_xticks(
            positions,
            [
                "FedAvg",
                "FedProx",
                "$\\lambda=0$",
                "Conflict",
                "Personal.",
                "FLEN",
                "Guarded-P",
                "Guarded-FLEN",
            ],
            rotation=20,
            ha="right",
        )
        axis.set_ylabel("Accuracy (%) $\\uparrow$")
        axis.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "e1_final_accuracy")


def plot_conflict_diagnostics(round_rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for method in (
        "Target cohort ($\\lambda=0$)",
        "Conflict only",
        "Personalization only",
        "FLEN",
        "Personalization guarded",
        "FLEN guarded",
    ):
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in round_rows:
            if row["method"] == method:
                grouped[int(row["round"])].append(row)
        rounds = sorted(grouped)
        exposure = [
            statistics.fmean(float(row["conflict_exposure_spread"]) for row in grouped[r])
            for r in rounds
        ]
        weight_delta = [
            statistics.fmean(float(row["max_weight_delta"]) for row in grouped[r])
            for r in rounds
        ]
        axes[0].plot(rounds, exposure, color=COLORS[method], lw=1.8, label=method)
        axes[1].plot(rounds, weight_delta, color=COLORS[method], lw=1.8, label=method)
    axes[0].set_title("Conflict exposure spread", loc="left", fontweight="bold")
    axes[1].set_title("Maximum weight change", loc="left", fontweight="bold")
    axes[0].set_ylabel("Max--min proxy exposure")
    axes[1].set_ylabel("$\\max_k |\\alpha_k-b_k|$")
    for axis in axes:
        axis.set_xlabel("Federated round")
        axis.set_xticks([1, 5, 10, 15, 20])
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_dir, "e1_conflict_diagnostics")


def render_report(
    summary_rows: list[dict[str, Any]],
    paired: dict[str, Any],
    local_residual: dict[str, Any],
) -> str:
    lines = [
        "# E1 CaseHOLD Federated Screening",
        "",
        "## Accepted Protocol",
        "",
        "All reported methods use 20 Flower rounds, four clients, seeds 41--43, "
        "1,200 training records, 240 validation records, and the same heuristic "
        "CaseHOLD partition protocol. The budget-matched unguarded personalization "
        "variants use eight shared plus two local steps; guarded variants retain ten "
        "shared steps and add two gated local steps, whose extra local compute is not "
        "included in communication cost.",
        "",
        "## Method Summary",
        "",
        "| Method | Aggregated accuracy | Mean client accuracy | Worst client accuracy | Communication (MiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | "
            f"{100 * row['server_final_accuracy_mean']:.2f} +/- "
            f"{100 * row['server_final_accuracy_sd']:.2f} | "
            f"{100 * row['client_final_accuracy_mean_mean']:.2f} +/- "
            f"{100 * row['client_final_accuracy_mean_sd']:.2f} | "
            f"{100 * row['client_final_accuracy_worst_mean']:.2f} +/- "
            f"{100 * row['client_final_accuracy_worst_sd']:.2f} | "
            f"{row['communication_mib_mean']:.2f} |"
        )
    lines.extend(["", "## Paired Component Effects", ""])
    for contrast_name, contrast in paired.items():
        lines.append(
            f"### {contrast_name}: {contrast['left_method']} minus "
            f"{contrast['right_method']}"
        )
        lines.append("")
        for metric, label in (
            ("server_final_accuracy", "Aggregated final accuracy"),
            ("client_final_accuracy_mean", "Mean client accuracy"),
            ("client_final_accuracy_worst", "Worst client accuracy"),
        ):
            item = contrast[metric]
            lower = 100 * (item["mean"] - item["ci95_half_width"])
            upper = 100 * (item["mean"] + item["ci95_half_width"])
            lines.append(
                f"- {label}: {100 * item['mean']:+.2f} percentage points "
                f"(95% paired t interval {lower:+.2f} to {upper:+.2f})."
            )
        lines.append("")
    lines.extend(["## Within-Client Local Residual Effect", ""])
    for method, item in local_residual.items():
        lower = 100 * (item["mean"] - item["ci95_half_width"])
        upper = 100 * (item["mean"] + item["ci95_half_width"])
        lines.append(
            f"- {method}: personalized minus shared-only accuracy is "
            f"{100 * item['mean']:+.2f} percentage points "
            f"(95% paired t interval {lower:+.2f} to {upper:+.2f})."
        )
    lines.append("")
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "This automatic-proxy pilot can support statements about runtime execution, "
            "optimization behavior, communication accounting, and whether the configured "
            "weighting mechanism activates. It cannot support claims that the proxy measures "
            "expert legal conflict or that heuristic CaseHOLD partitions represent authentic "
            "cross-jurisdiction transfer. Three seeds also provide low power; uncertainty "
            "intervals and all seed-level outcomes must accompany any comparative statement.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows, round_rows = collect_runs(input_root, args.seeds)
    summary_rows = aggregate_methods(seed_rows)
    paired = paired_effects(seed_rows)
    local_residual = paired_local_residual_effect(seed_rows)
    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "round_metrics.csv", round_rows)
    write_csv(output_dir / "method_summary.csv", summary_rows)
    (output_dir / "paired_flen_effects.json").write_text(
        json.dumps(
            {"component_contrasts": paired, "local_residual": local_residual},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "analysis_report.md").write_text(
        render_report(summary_rows, paired, local_residual),
        encoding="utf-8",
    )
    setup_style()
    plot_convergence(round_rows, output_dir)
    plot_final_summary(summary_rows, output_dir)
    plot_conflict_diagnostics(round_rows, output_dir)
    print(
        json.dumps(
            {
                "methods": summary_rows,
                "paired": paired,
                "local_residual": local_residual,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
