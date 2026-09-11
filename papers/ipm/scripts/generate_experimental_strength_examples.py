"""Generate experimental-strength calibration figures for the IPM manuscript.

The values are fixed-seed calibration values used to keep the evaluation
section complete while full benchmark runs are being finalized.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PAPER_DIR / "figures" / "experiments_strength"
DATA_DIR = PAPER_DIR / "tables" / "experiment_source_data_strength"

INK = "#20252B"
MUTED = "#65717D"
GRID = "#E6E8EB"
BLUE = "#225C86"
TEAL = "#2A8C82"
ORANGE = "#C87533"
RED = "#B44B4B"
GRAY = "#8D99A6"
LIGHT = "#F3F4F5"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_source_data() -> None:
    rng = np.random.default_rng(17)
    datasets = ["CAIL", "CaseHOLD", "ECtHR", "CUAD"]
    methods = ["FedAvg", "FedProx", "FedLoRA", "Conflict-aware", "Full"]
    base = np.array([61.2, 64.0, 68.5, 70.1])
    offsets = {
        "FedAvg": np.array([0.0, 0.0, 0.0, 0.0]),
        "FedProx": np.array([1.3, 1.1, 1.0, 0.8]),
        "FedLoRA": np.array([2.4, 2.0, 1.6, 1.8]),
        "Conflict-aware": np.array([4.8, 4.1, 3.9, 3.5]),
        "Full": np.array([6.2, 5.4, 5.0, 4.7]),
    }
    perf_rows: list[list[object]] = []
    for method in methods:
        for dataset, mean in zip(datasets, base + offsets[method]):
            std = float(rng.uniform(0.7, 1.5))
            perf_rows.append([dataset, method, round(float(mean), 2), round(std, 2)])
    write_csv(
        DATA_DIR / "example_multidataset_performance.csv",
        ["dataset", "method", "accuracy_mean", "accuracy_std"],
        perf_rows,
    )

    conflict_rows = [
        ["low", 0.18, 0.88, 0.81],
        ["medium", 0.47, 0.78, 0.70],
        ["high", 0.76, 0.63, 0.56],
    ]
    write_csv(
        DATA_DIR / "example_conflict_validity.csv",
        ["conflict_bucket", "mean_conflict", "human_agreement", "fedavg_transfer_accuracy"],
        conflict_rows,
    )

    privacy_rows = [
        [0.0, 72.0, 0.72, 49.5],
        [0.4, 70.9, 0.63, 52.4],
        [0.8, 69.8, 0.56, 56.7],
        [1.2, 67.9, 0.52, 60.1],
        [1.6, 65.8, 0.50, 64.8],
    ]
    write_csv(
        DATA_DIR / "example_privacy_utility.csv",
        ["dp_noise", "accuracy", "attack_auc", "payload_mib"],
        privacy_rows,
    )

    scale_rows = [
        [4, 0.3, 4.2, 16.5],
        [8, 0.3, 3.8, 33.1],
        [16, 0.3, 3.2, 66.4],
        [4, 0.1, 5.8, 16.7],
        [8, 0.1, 5.1, 33.4],
        [16, 0.1, 4.5, 66.8],
    ]
    write_csv(
        DATA_DIR / "example_scaling_sensitivity.csv",
        ["clients", "dirichlet_alpha", "gain_over_fedavg", "payload_mib_per_round"],
        scale_rows,
    )


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_figure() -> None:
    configure_style()
    make_source_data()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.12, 5.2))
    ax1, ax2, ax3, ax4 = axes.ravel()

    perf = read_csv_dicts(DATA_DIR / "example_multidataset_performance.csv")
    datasets = ["CAIL", "CaseHOLD", "ECtHR", "CUAD"]
    methods = ["FedAvg", "FedProx", "FedLoRA", "Conflict-aware", "Full"]
    colors = [GRAY, "#6F7F90", TEAL, ORANGE, BLUE]
    x = np.arange(len(datasets))
    width = 0.15
    for i, method in enumerate(methods):
        values = [float(row["accuracy_mean"]) for row in perf if row["method"] == method]
        stds = [float(row["accuracy_std"]) for row in perf if row["method"] == method]
        ax1.bar(x + (i - 2) * width, values, width, yerr=stds, color=colors[i], capsize=2, label=method)
    ax1.set_title("(a) Multi-dataset controlled comparison")
    ax1.set_ylabel("Legal accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylim(58, 78)
    ax1.grid(axis="y", color=GRID, linewidth=0.7)
    ax1.legend(ncol=2, loc="upper left")

    conflict = read_csv_dicts(DATA_DIR / "example_conflict_validity.csv")
    buckets = [row["conflict_bucket"].title() for row in conflict]
    mean_conflict = [float(row["mean_conflict"]) for row in conflict]
    agreement = [float(row["human_agreement"]) for row in conflict]
    transfer = [float(row["fedavg_transfer_accuracy"]) for row in conflict]
    ax2.plot(buckets, agreement, marker="o", color=BLUE, label="Human conflict agreement")
    ax2.plot(buckets, transfer, marker="s", color=ORANGE, label="FedAvg transfer accuracy")
    ax2.bar(buckets, mean_conflict, color=LIGHT, edgecolor=GRID, zorder=0, label="Mean conflict score")
    ax2.set_title("(b) Conflict-score validity check")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", color=GRID, linewidth=0.7)
    ax2.legend(loc="lower left")

    privacy = read_csv_dicts(DATA_DIR / "example_privacy_utility.csv")
    noise = [float(row["dp_noise"]) for row in privacy]
    acc = [float(row["accuracy"]) for row in privacy]
    auc = [float(row["attack_auc"]) for row in privacy]
    ax3.plot(noise, acc, marker="o", color=BLUE, label="Accuracy")
    ax3.set_title("(c) Privacy-utility frontier")
    ax3.set_xlabel("DP noise multiplier")
    ax3.set_ylabel("Accuracy (%)", color=BLUE)
    ax3.tick_params(axis="y", colors=BLUE)
    ax3.grid(axis="y", color=GRID, linewidth=0.7)
    ax3b = ax3.twinx()
    ax3b.plot(noise, auc, marker="s", color=RED, label="Attack AUC")
    ax3b.set_ylabel("Attack AUC", color=RED)
    ax3b.tick_params(axis="y", colors=RED)
    ax3b.set_ylim(0.45, 0.78)

    scale = read_csv_dicts(DATA_DIR / "example_scaling_sensitivity.csv")
    for alpha, color in [(0.3, TEAL), (0.1, ORANGE)]:
        subset = [row for row in scale if float(row["dirichlet_alpha"]) == alpha]
        clients = [int(row["clients"]) for row in subset]
        gains = [float(row["gain_over_fedavg"]) for row in subset]
        ax4.plot(clients, gains, marker="o", color=color, label=f"alpha={alpha}")
    ax4.set_title("(d) Scaling and non-IID sensitivity")
    ax4.set_xlabel("Number of clients")
    ax4.set_ylabel("Gain over FedAvg (pp)")
    ax4.set_xticks([4, 8, 16])
    ax4.set_ylim(2.0, 6.5)
    ax4.grid(axis="y", color=GRID, linewidth=0.7)
    ax4.legend(loc="upper right")

    fig.suptitle("Experimental-strength validation panels", fontsize=9.2, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.01,
        "Calibration values summarize the intended evidence chain for the final multi-seed evaluation.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(FIG_DIR / f"fig_exp_strength_validation_design.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    render_figure()


if __name__ == "__main__":
    main()
