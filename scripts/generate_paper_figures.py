"""Generate publication-style placeholder figures for the IPM manuscript.

The quantitative values in this script are placeholders. Replace the CSV files
written under outputs/figures/source_data with real experiment outputs before
making empirical claims.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures" / "paper"
DATA_DIR = ROOT / "outputs" / "figures" / "source_data"


COLORS = {
    "ink": "#27313F",
    "muted": "#697386",
    "line": "#B8C1CC",
    "panel": "#F7F9FC",
    "client": "#D9E8F6",
    "client_edge": "#6B98C4",
    "server": "#E6DFF2",
    "server_edge": "#8E75B6",
    "privacy": "#DCEFE5",
    "privacy_edge": "#5E9C78",
    "conflict": "#F7E0D4",
    "conflict_edge": "#C9805C",
    "agent": "#FFF2CC",
    "agent_edge": "#C49A3A",
    "accent": "#2F6F9F",
    "accent2": "#8E75B6",
    "accent3": "#C9805C",
    "good": "#4F9D69",
    "bad": "#C75D5D",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 400
        fig.savefig(FIG_DIR / f"{name}.{suffix}", **kwargs)
    plt.close(fig)


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = DATA_DIR / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    fc: str,
    ec: str,
    fontsize: float = 7.5,
    lw: float = 1.0,
    radius: float = 0.025,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        linespacing=1.18,
    )
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#7A8594",
    lw: float = 1.0,
    rad: float = 0.0,
    style: str = "-|>",
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        lw=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)


def panel_label(ax: plt.Axes, label: str, x: float = 0.0, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def placeholder_note(ax: plt.Axes, text: str = "placeholder values") -> None:
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=COLORS["muted"],
    )


def fig1_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "Federated legal LLM architecture",
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        0.02,
        0.925,
        "Raw judicial records remain local; only protected adapters, updates, or reasoning embeddings are exchanged.",
        fontsize=7.5,
        color=COLORS["muted"],
        va="top",
    )

    client_labels = [
        "Court CN\nCAIL / LeCaRD",
        "Court US\nCaseHOLD",
        "Institute EU\nECtHR",
        "Law firm\nCUAD",
    ]
    client_x = [0.06, 0.06, 0.06, 0.06]
    client_y = [0.72, 0.56, 0.40, 0.24]
    for x, y, label in zip(client_x, client_y, client_labels):
        box(ax, (x, y), (0.16, 0.095), label, fc=COLORS["client"], ec=COLORS["client_edge"], fontsize=6.8)
        ax.text(x + 0.085, y - 0.018, "private corpus + citation graph", ha="center", va="top", fontsize=5.8, color=COLORS["muted"])

    layer_boxes = [
        ("Local LoRA\ntraining", 0.31, 0.72, COLORS["panel"], COLORS["client_edge"], 0.17),
        ("Jurisdiction-aware\nrepresentation", 0.31, 0.54, COLORS["panel"], COLORS["accent"], 0.17),
        ("Privacy controls\nDP + secure aggregation", 0.31, 0.36, COLORS["privacy"], COLORS["privacy_edge"], 0.17),
        ("Conflict-aware\nserver aggregation", 0.56, 0.54, COLORS["server"], COLORS["server_edge"], 0.17),
        ("Federated multi-agent\njudicial reasoning", 0.78, 0.54, COLORS["agent"], COLORS["agent_edge"], 0.16),
    ]
    for label, x, y, fc, ec, w in layer_boxes:
        box(ax, (x, y), (w, 0.12), label, fc=fc, ec=ec, fontsize=6.8)

    for y in client_y:
        arrow(ax, (0.22, y + 0.047), (0.31, 0.78), rad=0.03, color=COLORS["client_edge"])
    arrow(ax, (0.395, 0.72), (0.395, 0.66), color=COLORS["accent"])
    arrow(ax, (0.395, 0.54), (0.395, 0.48), color=COLORS["privacy_edge"])
    arrow(ax, (0.48, 0.60), (0.56, 0.60), color=COLORS["server_edge"])
    arrow(ax, (0.73, 0.60), (0.78, 0.60), color=COLORS["agent_edge"])

    box(ax, (0.56, 0.30), (0.17, 0.105), "Evaluation\naccuracy | citations | drift", fc="#EEF3F9", ec=COLORS["line"], fontsize=7.0)
    box(ax, (0.78, 0.30), (0.16, 0.105), "Reproducibility\nconfigs | logs | figures", fc="#EEF3F9", ec=COLORS["line"], fontsize=6.8)
    arrow(ax, (0.645, 0.54), (0.645, 0.405), color=COLORS["line"])
    arrow(ax, (0.855, 0.54), (0.855, 0.405), color=COLORS["line"])

    ax.text(0.39, 0.88, "LoRA", ha="center", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.525, 0.665, "updates", ha="center", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.75, 0.695, "reasoning embeddings", ha="center", fontsize=6.0, color=COLORS["muted"])
    save_fig(fig, "fig1_framework")


def fig2_conflict_aggregation() -> None:
    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for a in (ax, ax2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")

    panel_label(ax, "a")
    ax.text(0.08, 0.96, "Aggregation logic", fontsize=10, fontweight="bold", color=COLORS["ink"], va="top")
    client_positions = [(0.05, 0.73), (0.05, 0.55), (0.05, 0.37), (0.05, 0.19)]
    labels = ["CN civil law", "US common law", "EU regulatory", "Contract law"]
    for (x, y), label in zip(client_positions, labels):
        box(ax, (x, y), (0.23, 0.10), label, fc=COLORS["client"], ec=COLORS["client_edge"], fontsize=6.6)
        arrow(ax, (x + 0.23, y + 0.05), (0.47, 0.54), color=COLORS["client_edge"], rad=0.05)

    box(ax, (0.47, 0.47), (0.22, 0.14), "Conflict-weighted\nadapter average", fc=COLORS["server"], ec=COLORS["server_edge"], fontsize=6.8)
    arrow(ax, (0.69, 0.54), (0.82, 0.54), color=COLORS["server_edge"])
    box(ax, (0.82, 0.47), (0.12, 0.14), "Global\nadapter", fc="#F1ECF8", ec=COLORS["server_edge"], fontsize=6.4)
    ax.text(0.42, 0.32, r"$\widetilde{\alpha}_k \propto p_k\exp(-\lambda\delta_k)$", fontsize=8.1, color=COLORS["ink"])
    ax.text(0.42, 0.25, r"$\Delta_{t+1}=\sum_k\widetilde{\alpha}_k\Delta_{t,k}$", fontsize=8.1, color=COLORS["ink"])
    ax.text(0.42, 0.18, "Conflict exposure downweights incompatible updates.", fontsize=6.4, color=COLORS["muted"])

    panel_label(ax2, "b")
    ax2.text(0.02, 0.96, "Conflict dimensions", fontsize=10, fontweight="bold", color=COLORS["ink"], va="top")
    conflict_boxes = [
        ("Citation\nconflict", 0.09, 0.72),
        ("Reasoning\nconflict", 0.59, 0.72),
        ("Verdict\nconflict", 0.09, 0.45),
        ("Rule-alignment\ndistance", 0.59, 0.45),
    ]
    for label, x, y in conflict_boxes:
        box(ax2, (x, y), (0.30, 0.12), label, fc=COLORS["conflict"], ec=COLORS["conflict_edge"])
        arrow(ax2, (x + 0.15, y), (0.50, 0.32), color=COLORS["conflict_edge"], rad=0.08)
    box(ax2, (0.34, 0.21), (0.32, 0.12), r"$D_{\mathrm{conflict}}$", fc="#F9E9E0", ec=COLORS["conflict_edge"], fontsize=8.5)
    ax2.text(
        0.50,
        0.08,
        r"$\omega_{\mathrm{cit}}D_{\mathrm{cit}}+\omega_{\mathrm{rea}}D_{\mathrm{rea}}+\omega_{\mathrm{ver}}D_{\mathrm{ver}}+\omega_{\mathrm{rule}}D_{\mathrm{rule}}$",
        ha="center",
        fontsize=6.4,
        color=COLORS["ink"],
    )
    save_fig(fig, "fig2_conflict_aggregation")


def fig3_multi_agent_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "Federated multi-agent judicial reasoning workflow", fontsize=11, fontweight="bold", color=COLORS["ink"], va="top")

    nodes = {
        "case": ("Case input\nfacts | issues | evidence", 0.04, 0.60, 0.17, 0.12, COLORS["panel"], COLORS["line"]),
        "align": ("Jurisdiction\nalignment", 0.28, 0.60, 0.16, 0.12, COLORS["agent"], COLORS["agent_edge"]),
        "local": ("Local institution\nreasoning", 0.52, 0.60, 0.16, 0.12, COLORS["client"], COLORS["client_edge"]),
        "pro": ("Prosecutor\nagent", 0.30, 0.36, 0.15, 0.105, COLORS["agent"], COLORS["agent_edge"]),
        "def": ("Defence\nagent", 0.52, 0.36, 0.15, 0.105, COLORS["agent"], COLORS["agent_edge"]),
        "cite": ("Citation\nverification", 0.73, 0.70, 0.16, 0.105, COLORS["privacy"], COLORS["privacy_edge"]),
        "conf": ("Conflict\ndetection", 0.73, 0.52, 0.16, 0.105, COLORS["conflict"], COLORS["conflict_edge"]),
        "audit": ("Privacy\nauditor", 0.73, 0.34, 0.16, 0.105, COLORS["privacy"], COLORS["privacy_edge"]),
        "judge": ("Judge aggregation\nfinal reasoning", 0.40, 0.12, 0.22, 0.12, COLORS["server"], COLORS["server_edge"]),
    }
    for label, x, y, w, h, fc, ec in nodes.values():
        box(ax, (x, y), (w, h), label, fc=fc, ec=ec, fontsize=7.2)

    arrow(ax, (0.21, 0.66), (0.28, 0.66), color=COLORS["line"])
    arrow(ax, (0.44, 0.66), (0.52, 0.66), color=COLORS["client_edge"])
    arrow(ax, (0.60, 0.60), (0.38, 0.465), color=COLORS["agent_edge"], rad=0.05)
    arrow(ax, (0.60, 0.60), (0.60, 0.465), color=COLORS["agent_edge"], rad=-0.05)
    arrow(ax, (0.68, 0.66), (0.73, 0.755), color=COLORS["privacy_edge"], rad=0.05)
    arrow(ax, (0.68, 0.66), (0.73, 0.575), color=COLORS["conflict_edge"], rad=-0.02)
    arrow(ax, (0.68, 0.66), (0.73, 0.395), color=COLORS["privacy_edge"], rad=-0.08)
    for x in (0.375, 0.595, 0.81):
        arrow(ax, (x, 0.36), (0.51, 0.24), color=COLORS["server_edge"], rad=0.05 if x < 0.55 else -0.05)

    ax.text(0.52, 0.78, "shared: abstract embeddings / verdict distributions", ha="center", fontsize=6.6, color=COLORS["muted"])
    ax.text(0.52, 0.54, "private: raw cases, citations, local knowledge bases", ha="center", fontsize=6.6, color=COLORS["muted"])
    save_fig(fig, "fig3_multi_agent_workflow")


def fig4_experiment_design() -> None:
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.12, 1.0], hspace=0.38, wspace=0.28)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    panel_label(ax1, "a")
    ax1.text(0.03, 0.96, "Experiment workflow", fontsize=10, fontweight="bold", va="top", color=COLORS["ink"])
    steps = [
        ("YAML config", 0.08, 0.78),
        ("Dataset adapters", 0.38, 0.78),
        ("Non-IID partitions", 0.68, 0.78),
        ("Federated rounds", 0.20, 0.48),
        ("Agent reasoning", 0.52, 0.48),
        ("Metrics + figures", 0.36, 0.20),
    ]
    for label, x, y in steps:
        box(ax1, (x, y), (0.22, 0.12), label, fc=COLORS["panel"], ec=COLORS["line"], fontsize=7.2)
    arrow(ax1, (0.30, 0.84), (0.38, 0.84), color=COLORS["line"])
    arrow(ax1, (0.60, 0.84), (0.68, 0.84), color=COLORS["line"])
    arrow(ax1, (0.79, 0.78), (0.31, 0.60), color=COLORS["line"], rad=0.12)
    arrow(ax1, (0.42, 0.54), (0.52, 0.54), color=COLORS["line"])
    arrow(ax1, (0.63, 0.48), (0.50, 0.32), color=COLORS["line"])
    arrow(ax1, (0.31, 0.48), (0.43, 0.32), color=COLORS["line"])

    panel_label(ax2, "b")
    clients = ["CN", "US", "EU", "Contract"]
    datasets = ["CAIL", "CaseHOLD", "ECtHR", "CUAD"]
    partition = np.array(
        [
            [0.72, 0.10, 0.08, 0.10],
            [0.08, 0.72, 0.10, 0.10],
            [0.08, 0.12, 0.70, 0.10],
            [0.10, 0.08, 0.10, 0.72],
        ]
    )
    im = ax2.imshow(partition, cmap="Blues", vmin=0, vmax=0.8)
    ax2.set_xticks(range(len(datasets)), datasets, rotation=35, ha="right")
    ax2.set_yticks(range(len(clients)), clients)
    ax2.set_title("Placeholder non-IID partition", fontsize=8, loc="left", pad=8)
    for i in range(partition.shape[0]):
        for j in range(partition.shape[1]):
            ax2.text(j, i, f"{partition[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("client share", fontsize=6.5)
    placeholder_note(ax2)

    panel_label(ax3, "c")
    rounds = np.array([1, 2, 3, 4, 5])
    payload = np.array([1.00, 0.78, 0.66, 0.58, 0.54])
    ax3.plot(rounds, payload, marker="o", color=COLORS["accent"], lw=1.6, ms=4)
    ax3.set_xlabel("federated round")
    ax3.set_ylabel("relative payload")
    ax3.set_ylim(0.45, 1.05)
    ax3.set_title("Communication logging slot", fontsize=8, loc="left", pad=8)
    ax3.grid(axis="y", color="#E8EDF3", lw=0.7)
    placeholder_note(ax3)

    rows = []
    for i, client in enumerate(clients):
        for j, dataset in enumerate(datasets):
            rows.append({"client": client, "dataset": dataset, "share": partition[i, j]})
    write_csv("fig4_partition_placeholder.csv", rows)
    write_csv("fig4_payload_placeholder.csv", [{"round": int(r), "relative_payload": float(v)} for r, v in zip(rounds, payload)])
    save_fig(fig, "fig4_experiment_design")


def fig5_placeholder_results() -> None:
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.42)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    methods = ["FedAvg", "FedProx", "Conflict", "Full"]
    x = np.arange(len(methods))

    accuracy = np.array([0.61, 0.63, 0.67, 0.70])
    citation = np.array([0.58, 0.61, 0.69, 0.74])
    hallucination = np.array([0.24, 0.22, 0.17, 0.14])
    rounds = np.arange(1, 11)
    comm = {
        "full fine-tune": 1.00 * np.ones_like(rounds, dtype=float),
        "LoRA": np.linspace(0.30, 0.24, len(rounds)),
        "LoRA + compression": np.linspace(0.18, 0.12, len(rounds)),
    }
    drift = {
        "FedAvg": np.linspace(0.34, 0.27, len(rounds)) + 0.015 * np.sin(rounds),
        "Conflict-aware": np.linspace(0.30, 0.18, len(rounds)) + 0.010 * np.cos(rounds),
    }

    ax = axes[0]
    panel_label(ax, "a", x=-0.12, y=1.08)
    ax.bar(x, accuracy, color=[COLORS["line"], COLORS["line"], COLORS["accent2"], COLORS["accent"]], edgecolor="white")
    ax.set_xticks(x, methods, rotation=20, ha="right")
    ax.set_ylim(0.50, 0.76)
    ax.set_ylabel("legal accuracy")
    ax.set_title("Main comparison slot", fontsize=8, loc="left", pad=12)
    ax.grid(axis="y", color="#E8EDF3", lw=0.7)
    placeholder_note(ax)

    ax = axes[1]
    panel_label(ax, "b", x=-0.12, y=1.08)
    width = 0.36
    ax.bar(x - width / 2, citation, width=width, color=COLORS["good"], label="citation consistency")
    ax.bar(x + width / 2, hallucination, width=width, color=COLORS["bad"], label="hallucination rate")
    ax.set_xticks(x, methods, rotation=20, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_title("Legal reliability metrics", fontsize=8, loc="left", pad=12)
    ax.legend(fontsize=6.4, loc="upper left")
    ax.grid(axis="y", color="#E8EDF3", lw=0.7)
    placeholder_note(ax)

    ax = axes[2]
    panel_label(ax, "c", x=-0.12, y=1.08)
    for label, values in comm.items():
        color = COLORS["line"] if label == "full fine-tune" else COLORS["accent2"] if label == "LoRA" else COLORS["accent"]
        ax.plot(rounds, values, marker="o", lw=1.4, ms=3, label=label, color=color)
    ax.set_xlabel("federated round")
    ax.set_ylabel("relative communication")
    ax.set_ylim(0, 1.08)
    ax.set_title("Communication-cost slot", fontsize=8, loc="left", pad=12)
    ax.legend(fontsize=6.4)
    ax.grid(axis="y", color="#E8EDF3", lw=0.7)
    placeholder_note(ax)

    ax = axes[3]
    panel_label(ax, "d", x=-0.12, y=1.08)
    for label, values in drift.items():
        color = COLORS["line"] if label == "FedAvg" else COLORS["conflict_edge"]
        ax.plot(rounds, values, marker="o", lw=1.4, ms=3, label=label, color=color)
    ax.set_xlabel("federated round")
    ax.set_ylabel("client drift")
    ax.set_ylim(0.12, 0.38)
    ax.set_title("Client-drift slot", fontsize=8, loc="left", pad=12)
    ax.legend(fontsize=6.4)
    ax.grid(axis="y", color="#E8EDF3", lw=0.7)
    placeholder_note(ax)

    rows = []
    for i, method in enumerate(methods):
        rows.append(
            {
                "method": method,
                "legal_accuracy": float(accuracy[i]),
                "citation_consistency": float(citation[i]),
                "hallucination_rate": float(hallucination[i]),
            }
        )
    write_csv("fig5_method_metrics_placeholder.csv", rows)
    comm_rows = []
    for label, values in comm.items():
        for r, value in zip(rounds, values):
            comm_rows.append({"method": label, "round": int(r), "relative_communication": float(value)})
    write_csv("fig5_communication_placeholder.csv", comm_rows)
    drift_rows = []
    for label, values in drift.items():
        for r, value in zip(rounds, values):
            drift_rows.append({"method": label, "round": int(r), "client_drift": float(value)})
    write_csv("fig5_client_drift_placeholder.csv", drift_rows)
    save_fig(fig, "fig5_placeholder_results")


def main() -> None:
    setup_style()
    ensure_dirs()
    fig1_framework()
    fig2_conflict_aggregation()
    fig3_multi_agent_workflow()
    fig4_experiment_design()
    fig5_placeholder_results()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote placeholder source data to {DATA_DIR}")


if __name__ == "__main__":
    main()
