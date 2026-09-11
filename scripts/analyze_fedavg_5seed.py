"""Validate and analyze the completed five-seed CaseHOLD FedAvg run."""

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
COLORS = {
    "blue": "#2F6F9F",
    "orange": "#C9805C",
    "green": "#4F9D69",
    "purple": "#8E75B6",
    "gray": "#697386",
    "light": "#D9E8F6",
    "grid": "#E8EDF3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="outputs/experiment_analysis/fedavg_5seed/raw",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/experiment_analysis/fedavg_5seed/results",
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_sd_ci(values: Iterable[float]) -> tuple[float, float, float]:
    data = [float(value) for value in values]
    mean = statistics.fmean(data)
    if len(data) < 2:
        return mean, 0.0, 0.0
    sd = statistics.stdev(data)
    critical = T_CRITICAL_95.get(len(data), 1.96)
    return mean, sd, critical * sd / math.sqrt(len(data))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_and_collect(
    input_root: Path,
    expected_seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None

    for seed in expected_seeds:
        seed_dir = input_root / f"seed_{seed}"
        required = [
            "real_experiment_report.json",
            "server_eval_rounds.jsonl",
            "client_eval_log.jsonl",
            "communication_log.jsonl",
            "resolved_config.json",
        ]
        missing = [name for name in required if not (seed_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Seed {seed} is missing: {', '.join(missing)}")

        for path in sorted(seed_dir.glob("*")):
            if path.is_file():
                manifest_rows.append(
                    {
                        "seed": seed,
                        "path": path.relative_to(input_root).as_posix(),
                        "bytes": path.stat().st_size,
                    }
                )

        report = read_json(seed_dir / "real_experiment_report.json")
        config = read_json(seed_dir / "resolved_config.json")
        server_eval = read_jsonl(seed_dir / "server_eval_rounds.jsonl")
        client_eval = read_jsonl(seed_dir / "client_eval_log.jsonl")
        communication = read_jsonl(seed_dir / "communication_log.jsonl")

        if int(report["seed"]) != seed or not report.get("real_training"):
            raise ValueError(f"Seed {seed} report is not a matching real-training run")
        comparable_config = {key: value for key, value in config.items() if key not in {"seed", "run_name"}}
        if reference_config is None:
            reference_config = comparable_config
        elif comparable_config != reference_config:
            raise ValueError(f"Seed {seed} uses a different experiment configuration")

        num_rounds = int(report["num_rounds"])
        num_clients = int(report["num_clients"])
        expected_pairs = {(round_id, f"client_{client_id}") for round_id in range(1, num_rounds + 1) for client_id in range(num_clients)}
        eval_pairs = {(int(row["metrics"]["server_round"]), row["client_id"]) for row in client_eval}
        comm_pairs = {(int(row["metrics"]["server_round"]), row["client_id"]) for row in communication}
        if len(server_eval) != num_rounds or eval_pairs != expected_pairs or comm_pairs != expected_pairs:
            raise ValueError(f"Seed {seed} has incomplete round/client logs")
        if any(int(row["num_failures"]) != 0 or int(row["num_results"]) != num_clients for row in server_eval):
            raise ValueError(f"Seed {seed} contains failed or missing client evaluations")

        final_eval = max(server_eval, key=lambda row: int(row["round"]))
        final_accuracy = float(final_eval["metrics"]["accuracy"])
        if not math.isclose(final_accuracy, float(report["final_accuracy"]), abs_tol=1e-12):
            raise ValueError(f"Seed {seed} report and final-round accuracy disagree")
        communication_bytes = sum(int(row["num_bytes"]) for row in communication)
        if communication_bytes != int(report["communication_bytes"]):
            raise ValueError(f"Seed {seed} communication totals disagree")

        seed_rows.append(
            {
                "seed": seed,
                "final_accuracy": final_accuracy,
                "best_accuracy": float(report["best_accuracy"]),
                "mean_train_loss": float(report["mean_train_loss"]),
                "mean_client_drift_l2": float(report["mean_client_drift_l2"]),
                "communication_bytes": communication_bytes,
                "communication_mib": communication_bytes / MIB,
                "eval_samples": int(report["max_eval_samples"]),
                "rounds": num_rounds,
                "clients": num_clients,
            }
        )

        comm_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in communication:
            comm_by_round[int(row["metrics"]["server_round"])].append(row)
        for row in sorted(server_eval, key=lambda item: int(item["round"])):
            round_id = int(row["round"])
            comm_rows = comm_by_round[round_id]
            round_rows.append(
                {
                    "seed": seed,
                    "round": round_id,
                    "accuracy": float(row["metrics"]["accuracy"]),
                    "eval_loss": float(row["metrics"]["eval_loss"]),
                    "correct": int(row["metrics"]["correct"]),
                    "num_examples": int(row["metrics"]["num_examples"]),
                    "mean_client_drift_l2": statistics.fmean(float(item["metrics"]["client_drift_l2"]) for item in comm_rows),
                    "mean_train_loss": statistics.fmean(float(item["metrics"]["train_loss"]) for item in comm_rows),
                    "round_communication_mib": sum(int(item["num_bytes"]) for item in comm_rows) / MIB,
                }
            )
        for row in client_eval:
            metrics = row["metrics"]
            client_rows.append(
                {
                    "seed": seed,
                    "round": int(metrics["server_round"]),
                    "client_id": row["client_id"],
                    "accuracy": float(metrics["accuracy"]),
                    "eval_loss": float(metrics["eval_loss"]),
                    "correct": int(metrics["correct"]),
                    "num_examples": int(metrics["num_examples"]),
                }
            )

    return seed_rows, round_rows, client_rows, manifest_rows


def aggregate_rounds(round_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in round_rows:
        grouped[int(row["round"])].append(row)
    results = []
    for round_id, rows in sorted(grouped.items()):
        result: dict[str, Any] = {"round": round_id, "seeds": len(rows)}
        for metric in ("accuracy", "eval_loss", "mean_client_drift_l2", "mean_train_loss", "round_communication_mib"):
            mean, sd, ci95 = mean_sd_ci(float(row[metric]) for row in rows)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
            result[f"{metric}_ci95"] = ci95
        result["cumulative_communication_mib_mean"] = sum(
            item["round_communication_mib_mean"] for item in results
        ) + result["round_communication_mib_mean"]
        results.append(result)
    return results


def aggregate_clients(client_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_round = max(int(row["round"]) for row in client_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in client_rows:
        if int(row["round"]) == final_round:
            grouped[str(row["client_id"])].append(row)
    results = []
    for client_id, rows in sorted(grouped.items()):
        accuracy_mean, accuracy_sd, accuracy_ci95 = mean_sd_ci(float(row["accuracy"]) for row in rows)
        loss_mean, loss_sd, loss_ci95 = mean_sd_ci(float(row["eval_loss"]) for row in rows)
        results.append(
            {
                "client_id": client_id,
                "round": final_round,
                "seeds": len(rows),
                "accuracy_mean": accuracy_mean,
                "accuracy_sd": accuracy_sd,
                "accuracy_ci95": accuracy_ci95,
                "eval_loss_mean": loss_mean,
                "eval_loss_sd": loss_sd,
                "eval_loss_ci95": loss_ci95,
            }
        )
    return results


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(output_dir / f"{name}.{suffix}", **kwargs)
    plt.close(fig)


def plot_convergence(round_rows: list[dict[str, Any]], round_summary: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in round_rows:
        by_seed[int(row["seed"])].append(row)
    rounds = np.array([int(row["round"]) for row in round_summary])
    for seed, rows in sorted(by_seed.items()):
        ordered = sorted(rows, key=lambda row: int(row["round"]))
        axes[0].plot(rounds, [100 * float(row["accuracy"]) for row in ordered], color=COLORS["light"], lw=1.0, marker="o", ms=2.5)
        axes[1].plot(rounds, [float(row["eval_loss"]) for row in ordered], color="#EAD8CE", lw=1.0, marker="o", ms=2.5)
    accuracy_mean = np.array([100 * float(row["accuracy_mean"]) for row in round_summary])
    accuracy_sd = np.array([100 * float(row["accuracy_sd"]) for row in round_summary])
    loss_mean = np.array([float(row["eval_loss_mean"]) for row in round_summary])
    loss_sd = np.array([float(row["eval_loss_sd"]) for row in round_summary])
    axes[0].plot(rounds, accuracy_mean, color=COLORS["blue"], lw=2.2, marker="o", ms=4.5, label="Mean across 5 seeds")
    axes[0].fill_between(rounds, accuracy_mean - accuracy_sd, accuracy_mean + accuracy_sd, color=COLORS["blue"], alpha=0.16, label="±1 SD")
    axes[0].axhline(20.0, color=COLORS["gray"], ls="--", lw=1.0, label="5-way chance")
    axes[1].plot(rounds, loss_mean, color=COLORS["orange"], lw=2.2, marker="o", ms=4.5, label="Mean across 5 seeds")
    axes[1].fill_between(rounds, loss_mean - loss_sd, loss_mean + loss_sd, color=COLORS["orange"], alpha=0.16, label="±1 SD")
    axes[0].set_ylabel("Validation accuracy (%) ↑")
    axes[1].set_ylabel("Validation loss ↓")
    for axis, title in zip(axes, ("Accuracy convergence", "Loss convergence")):
        axis.set_xlabel("Federated round")
        axis.set_xticks(rounds)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color=COLORS["grid"], lw=0.8)
        axis.legend(fontsize=7)
    fig.suptitle("CaseHOLD FedAvg: five-seed convergence", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "fedavg_5seed_convergence")


def plot_seed_and_client_accuracy(
    seed_rows: list[dict[str, Any]],
    client_rows: list[dict[str, Any]],
    client_summary: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    seeds = [int(row["seed"]) for row in seed_rows]
    final_accuracy = np.array([100 * float(row["final_accuracy"]) for row in seed_rows])
    final_mean, _, final_ci = mean_sd_ci(final_accuracy)
    axes[0].bar([str(seed) for seed in seeds], final_accuracy, color=COLORS["light"], edgecolor=COLORS["blue"], linewidth=0.8)
    axes[0].axhline(final_mean, color=COLORS["blue"], lw=1.8, label=f"Mean {final_mean:.2f}%")
    axes[0].fill_between([-0.5, len(seeds) - 0.5], final_mean - final_ci, final_mean + final_ci, color=COLORS["blue"], alpha=0.14, label="95% CI")
    axes[0].axhline(20.0, color=COLORS["gray"], ls="--", lw=1.0, label="5-way chance")
    axes[0].set_xlabel("Random seed")
    axes[0].set_ylabel("Final validation accuracy (%) ↑")
    axes[0].set_title("Seed-level stability", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7)

    client_ids = [str(row["client_id"]) for row in client_summary]
    means = np.array([100 * float(row["accuracy_mean"]) for row in client_summary])
    sds = np.array([100 * float(row["accuracy_sd"]) for row in client_summary])
    positions = np.arange(len(client_ids))
    axes[1].bar(positions, means, yerr=sds, capsize=3, color="#E6DFF2", edgecolor=COLORS["purple"], linewidth=0.8, label="Mean ± SD")
    final_round = max(int(row["round"]) for row in client_rows)
    for seed_index, seed in enumerate(seeds):
        rows = [row for row in client_rows if int(row["seed"]) == seed and int(row["round"]) == final_round]
        values = {str(row["client_id"]): 100 * float(row["accuracy"]) for row in rows}
        jitter = (seed_index - 2) * 0.035
        axes[1].scatter(positions + jitter, [values[client_id] for client_id in client_ids], s=16, color=COLORS["purple"], alpha=0.72, zorder=3)
    axes[1].set_xticks(positions, client_ids)
    axes[1].set_xlabel("Federated client")
    axes[1].set_ylabel("Round-3 accuracy (%) ↑")
    axes[1].set_title("Client-level heterogeneity", loc="left", fontweight="bold")
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.grid(axis="y", color=COLORS["grid"], lw=0.8)
    fig.suptitle("CaseHOLD FedAvg: robustness and client disparity", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "fedavg_5seed_robustness")


def plot_efficiency(round_summary: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    rounds = np.array([int(row["round"]) for row in round_summary])
    drift_mean = np.array([float(row["mean_client_drift_l2_mean"]) for row in round_summary])
    drift_sd = np.array([float(row["mean_client_drift_l2_sd"]) for row in round_summary])
    cumulative_comm = np.array([float(row["cumulative_communication_mib_mean"]) for row in round_summary])
    axes[0].plot(rounds, drift_mean, color=COLORS["purple"], marker="o", ms=4.5, lw=2.0)
    axes[0].fill_between(rounds, drift_mean - drift_sd, drift_mean + drift_sd, color=COLORS["purple"], alpha=0.16, label="±1 SD")
    axes[0].set_ylabel("Mean update norm L2")
    axes[0].set_title("Update norm by round", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[1].plot(rounds, cumulative_comm, color=COLORS["green"], marker="o", ms=4.5, lw=2.0)
    axes[1].fill_between(rounds, 0, cumulative_comm, color=COLORS["green"], alpha=0.12)
    axes[1].set_ylabel("Cumulative communication (MiB) ↑")
    axes[1].set_title("Bidirectional payload", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("Federated round")
        axis.set_xticks(rounds)
        axis.grid(axis="y", color=COLORS["grid"], lw=0.8)
    fig.suptitle("CaseHOLD FedAvg: optimization and communication", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "fedavg_5seed_efficiency")


def build_summary(
    seed_rows: list[dict[str, Any]],
    round_summary: list[dict[str, Any]],
    client_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    final_mean, final_sd, final_ci95 = mean_sd_ci(float(row["final_accuracy"]) for row in seed_rows)
    train_loss_mean, train_loss_sd, train_loss_ci95 = mean_sd_ci(float(row["mean_train_loss"]) for row in seed_rows)
    drift_mean, drift_sd, drift_ci95 = mean_sd_ci(float(row["mean_client_drift_l2"]) for row in seed_rows)
    communication_mean, communication_sd, communication_ci95 = mean_sd_ci(float(row["communication_mib"]) for row in seed_rows)
    first_round = round_summary[0]
    final_round = round_summary[-1]
    client_accuracies = [float(row["accuracy_mean"]) for row in client_summary]
    return {
        "status": "passed",
        "completed_seeds": len(seed_rows),
        "expected_seeds": [int(row["seed"]) for row in seed_rows],
        "rounds_per_seed": int(seed_rows[0]["rounds"]),
        "clients_per_seed": int(seed_rows[0]["clients"]),
        "evaluation_samples_per_seed": int(seed_rows[0]["eval_samples"]),
        "final_accuracy": {"mean": final_mean, "sd": final_sd, "ci95_half_width": final_ci95},
        "round1_to_final_accuracy_gain": float(final_round["accuracy_mean"]) - float(first_round["accuracy_mean"]),
        "round1_to_final_eval_loss_reduction_fraction": 1.0 - float(final_round["eval_loss_mean"]) / float(first_round["eval_loss_mean"]),
        "mean_train_loss": {"mean": train_loss_mean, "sd": train_loss_sd, "ci95_half_width": train_loss_ci95},
        "mean_client_drift_l2": {"mean": drift_mean, "sd": drift_sd, "ci95_half_width": drift_ci95},
        "communication_mib": {"mean": communication_mean, "sd": communication_sd, "ci95_half_width": communication_ci95},
        "final_client_accuracy_gap": max(client_accuracies) - min(client_accuracies),
    }


def render_report(summary: dict[str, Any], round_summary: list[dict[str, Any]], client_summary: list[dict[str, Any]]) -> str:
    accuracy = summary["final_accuracy"]
    communication = summary["communication_mib"]
    lines = [
        "# Five-Seed CaseHOLD FedAvg Analysis",
        "",
        "## Integrity Audit",
        "",
        f"All {summary['completed_seeds']} expected seeds contain three server rounds, four client updates and four client evaluations per round. The logs report zero failed evaluations, and report-level totals match the raw round logs.",
        "",
        "## Main Results",
        "",
        f"Final validation accuracy is **{100 * accuracy['mean']:.2f}% +/- {100 * accuracy['sd']:.2f} SD** (95% t interval: {100 * (accuracy['mean'] - accuracy['ci95_half_width']):.2f}% to {100 * (accuracy['mean'] + accuracy['ci95_half_width']):.2f}%). Accuracy improves by {100 * summary['round1_to_final_accuracy_gain']:.2f} percentage points from round 1 to round 3, while evaluation loss falls by {100 * summary['round1_to_final_eval_loss_reduction_fraction']:.1f}%.",
        "",
        f"Each seed communicates {communication['mean']:.2f} MiB bidirectionally across three rounds. Mean client update norm relative to the broadcast adapter is {summary['mean_client_drift_l2']['mean']:.4f}; its non-monotonic trajectory should be treated as a FedAvg diagnostic rather than the pairwise client-drift metric required by the full protocol.",
        "",
        f"The round-3 gap between the strongest and weakest client mean accuracy is {100 * summary['final_client_accuracy_gap']:.2f} percentage points, showing material client-level heterogeneity despite a stable seed-level aggregate.",
        "",
        "## Round-Level Statistics",
        "",
        "| Round | Accuracy mean +/- SD | Eval loss mean +/- SD | Update norm mean +/- SD | Cumulative MiB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in round_summary:
        lines.append(
            f"| {row['round']} | {100 * row['accuracy_mean']:.2f} +/- {100 * row['accuracy_sd']:.2f}% | "
            f"{row['eval_loss_mean']:.4f} +/- {row['eval_loss_sd']:.4f} | "
            f"{row['mean_client_drift_l2_mean']:.4f} +/- {row['mean_client_drift_l2_sd']:.4f} | "
            f"{row['cumulative_communication_mib_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Final Client Statistics",
            "",
            "| Client | Accuracy mean +/- SD | Eval loss mean +/- SD |",
            "|---|---:|---:|",
        ]
    )
    for row in client_summary:
        lines.append(
            f"| {row['client_id']} | {100 * row['accuracy_mean']:.2f} +/- {100 * row['accuracy_sd']:.2f}% | "
            f"{row['eval_loss_mean']:.4f} +/- {row['eval_loss_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reviewer-Facing Limitations",
            "",
            "This is a bounded FedAvg sanity experiment, not a complete validation of the paper. It uses one CaseHOLD validation task, 240 evaluation examples, three rounds and ten local steps. It does not compare FedProx, SCAFFOLD, FedNova or conflict-aware aggregation, and it does not activate differential privacy, secure aggregation or multi-agent reasoning. The results therefore support pipeline functionality and basic convergence only.",
            "",
            "## Reproduction",
            "",
            "`python scripts/analyze_fedavg_5seed.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows, round_rows, client_rows, manifest_rows = validate_and_collect(input_root, args.expected_seeds)
    round_summary = aggregate_rounds(round_rows)
    client_summary = aggregate_clients(client_rows)
    summary = build_summary(seed_rows, round_summary, client_summary)

    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "round_metrics_raw.csv", round_rows)
    write_csv(output_dir / "round_metrics_summary.csv", round_summary)
    write_csv(output_dir / "client_round_metrics.csv", client_rows)
    write_csv(output_dir / "client_final_summary.csv", client_summary)
    write_csv(output_dir / "input_manifest.csv", manifest_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "analysis_report.md").write_text(render_report(summary, round_summary, client_summary), encoding="utf-8")

    setup_style()
    plot_convergence(round_rows, round_summary, output_dir)
    plot_seed_and_client_accuracy(seed_rows, client_rows, client_summary, output_dir)
    plot_efficiency(round_summary, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
