"""Generate publication-style experimental figure templates for the IPM paper.

This module is intentionally data-driven. Without ``--use-existing-data`` it
creates fixed-seed synthetic CSV files so that the manuscript layout can be
reviewed before experiments finish. Replace the CSV values with measured
outputs, then render with ``--use-existing-data --final-data``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

PAPER_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PAPER_DIR / "figures" / "experiments_topconf"
DATA_DIR = PAPER_DIR / "tables" / "experiment_source_data_topconf"

WIDTH = 7.12
METHODS = [
    "Centralized",
    "Local-only",
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "FedNova",
    "FedLoRA",
    "Conflict-aware",
    "Full framework",
]
FED_METHODS = [
    "FedAvg",
    "FedProx",
    "SCAFFOLD",
    "FedNova",
    "FedLoRA",
    "Conflict-aware",
    "Full framework",
]
CORE_TRANSFER_METHODS = [
    "Local-only",
    "Centralized",
    "FedAvg",
    "FedProx",
    "FedLoRA",
    "Conflict-aware",
    "Full framework",
]
DATASETS = ["CAIL", "CaseHOLD", "ECtHR", "CUAD"]
JURISDICTIONS = ["CN civil", "US common", "EU rights", "Contracts"]
SEEDS = [1, 2, 3, 4, 5]

COLORS = {
    "Centralized": "#626B75",
    "Local-only": "#C2C6CB",
    "FedAvg": "#AEB6BF",
    "FedProx": "#8D99A6",
    "SCAFFOLD": "#707E8E",
    "FedNova": "#536374",
    "FedLoRA": "#3F8584",
    "Conflict-aware": "#D27454",
    "Full framework": "#175A8B",
    "ink": "#20252B",
    "muted": "#65717D",
    "grid": "#E6E8EB",
    "light": "#F3F4F5",
    "placeholder": "#A94442",
}
LINESTYLES = {
    "FedAvg": (0, (3, 2)),
    "FedProx": (0, (2, 2)),
    "SCAFFOLD": (0, (5, 2)),
    "FedNova": (0, (1, 1)),
    "FedLoRA": (0, (4, 1, 1, 1)),
    "Conflict-aware": "-",
    "Full framework": "-",
}
HIGHLIGHTS = {"Conflict-aware", "Full framework"}
PLACEHOLDER_MODE = True


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.edgecolor": COLORS["ink"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    if grid:
        ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def figure_note(fig: plt.Figure) -> None:
    if not PLACEHOLDER_MODE:
        return
    fig.text(
        0.995,
        0.006,
        "SYNTHETIC TEMPLATE DATA | REPLACE WITH VERIFIED RUNS BEFORE SUBMISSION",
        ha="right",
        va="bottom",
        color=COLORS["placeholder"],
        fontsize=5.2,
        fontweight="bold",
    )


def save(fig: plt.Figure, name: str) -> None:
    figure_note(fig)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def mean_ci(
    frame: pd.DataFrame, group_cols: list[str], value_col: str
) -> pd.DataFrame:
    result = (
        frame.groupby(group_cols, sort=False)[value_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    result["ci"] = 1.96 * result["sd"].fillna(0) / np.sqrt(result["n"])
    return result


def load_or_generate(
    filename: str, generator, rng: np.random.Generator, use_existing: bool
) -> pd.DataFrame:
    path = DATA_DIR / filename
    if use_existing:
        if not path.exists():
            raise FileNotFoundError(f"Expected source data at {path}")
        return pd.read_csv(path)
    data = generator(rng)
    data.to_csv(path, index=False)
    return data


def draw_method_legend(fig: plt.Figure, methods: list[str], y: float = 0.99) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=4.2,
            lw=1.4 if method in HIGHLIGHTS else 1.0,
            linestyle=LINESTYLES.get(method, "-"),
            color=COLORS[method],
            label=method,
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, y),
        ncol=min(len(methods), 5),
        handletextpad=0.45,
        columnspacing=0.95,
    )


def generate_benchmark(rng: np.random.Generator) -> pd.DataFrame:
    base = {
        "Centralized": 0.684,
        "Local-only": 0.606,
        "FedAvg": 0.639,
        "FedProx": 0.651,
        "SCAFFOLD": 0.662,
        "FedNova": 0.658,
        "FedLoRA": 0.679,
        "Conflict-aware": 0.701,
        "Full framework": 0.719,
    }
    shifts = {"CAIL": -0.010, "CaseHOLD": 0.008, "ECtHR": -0.024, "CUAD": 0.015}
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for dataset in DATASETS:
            for method in METHODS:
                legal_accuracy = base[method] + shifts[dataset] + rng.normal(0, 0.007)
                citation_f1 = base[method] + 0.045 + shifts[dataset] / 2 + rng.normal(0, 0.008)
                coherence = base[method] + 0.030 + shifts[dataset] / 2 + rng.normal(0, 0.008)
                hallucination = 0.296 - (base[method] - 0.606) * 0.72 + rng.normal(0, 0.008)
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "method": method,
                        "legal_accuracy": np.clip(legal_accuracy, 0.50, 0.85),
                        "citation_f1": np.clip(citation_f1, 0.50, 0.90),
                        "reasoning_coherence": np.clip(coherence, 0.50, 0.90),
                        "hallucination_rate": np.clip(hallucination, 0.05, 0.40),
                    }
                )
    return pd.DataFrame(rows)


def plot_benchmark(data: pd.DataFrame) -> None:
    summary = mean_ci(data, ["dataset", "method"], "legal_accuracy")
    means = summary.pivot(index="dataset", columns="method", values="mean").loc[DATASETS, METHODS]
    aggregate = mean_ci(data, ["method"], "legal_accuracy").set_index("method").loc[METHODS]
    delta = means.subtract(means["FedAvg"], axis=0).T.drop(index="FedAvg")

    fig = plt.figure(figsize=(WIDTH, 3.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 0.88, 1.02], wspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    x = np.arange(len(DATASETS))
    for method in METHODS:
        subset = summary[summary["method"] == method].set_index("dataset").loc[DATASETS]
        ax_a.errorbar(
            x,
            subset["mean"],
            yerr=subset["ci"],
            marker="o",
            ms=3.2 if method not in HIGHLIGHTS else 4.0,
            lw=0.9 if method not in HIGHLIGHTS else 1.5,
            capsize=1.6,
            elinewidth=0.6,
            linestyle=LINESTYLES.get(method, "-"),
            color=COLORS[method],
            zorder=4 if method in HIGHLIGHTS else 2,
        )
    ax_a.set_xticks(x, DATASETS)
    ax_a.set_ylim(0.56, 0.77)
    ax_a.set_ylabel("Legal accuracy")
    ax_a.set_title("Task performance", loc="left")
    clean_axis(ax_a)
    panel(ax_a, "a")

    order = aggregate["mean"].sort_values().index.tolist()
    y = np.arange(len(order))
    ax_b.axvline(aggregate.loc["FedAvg", "mean"], color=COLORS["grid"], lw=0.9)
    for position, method in zip(y, order):
        point = aggregate.loc[method]
        ax_b.errorbar(
            point["mean"],
            position,
            xerr=point["ci"],
            fmt="o",
            ms=4.0 if method in HIGHLIGHTS else 3.2,
            lw=1.0,
            capsize=1.6,
            color=COLORS[method],
            zorder=3,
        )
    ax_b.set_yticks(y, order)
    ax_b.set_xlim(0.58, 0.75)
    ax_b.set_xlabel("Mean accuracy")
    ax_b.set_title("Across datasets", loc="left")
    ax_b.grid(axis="x", color=COLORS["grid"], lw=0.55)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    panel(ax_b, "b")

    cmap = LinearSegmentedColormap.from_list("delta", ["#B65D4E", "#F7F7F7", "#175A8B"])
    bound = float(np.abs(delta.values).max())
    im = ax_c.imshow(
        delta.values,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
    )
    ax_c.set_xticks(np.arange(len(DATASETS)), DATASETS, rotation=34, ha="right")
    ax_c.set_yticks(np.arange(len(delta.index)), delta.index)
    ax_c.set_title("Gain over FedAvg", loc="left")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax_c.text(j, i, f"{delta.iloc[i, j]*100:+.1f}", ha="center", va="center", fontsize=5.7)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_c, shrink=0.68, pad=0.02)
    cbar.ax.set_ylabel("points", rotation=90, labelpad=4)
    cbar.ax.tick_params(labelsize=5.8)
    panel(ax_c, "c")
    draw_method_legend(fig, METHODS, y=1.06)
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.98)
    save(fig, "fig_exp01_benchmark")


def plot_reliability(data: pd.DataFrame) -> None:
    metrics = [
        ("citation_f1", "Citation F1", True),
        ("reasoning_coherence", "Coherence", True),
        ("hallucination_rate", "Hallucination rate", False),
    ]
    summary = {
        metric: mean_ci(data, ["method"], metric).set_index("method").loc[METHODS]
        for metric, _, _ in metrics
    }
    matrix = np.column_stack([summary[metric]["mean"].values for metric, _, _ in metrics])

    fig = plt.figure(figsize=(WIDTH, 2.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=0.43)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    cmap = LinearSegmentedColormap.from_list("reliability", ["#F3F4F5", "#95B9C9", "#175A8B"])
    display = matrix.copy()
    display[:, 2] = 1 - display[:, 2]
    im = ax_a.imshow(display, cmap=cmap, aspect="auto", vmin=0.55, vmax=0.88)
    ax_a.set_yticks(np.arange(len(METHODS)), METHODS)
    ax_a.set_xticks(np.arange(3), ["Citation F1\nhigher", "Coherence\nhigher", "Hallucination\nlower"])
    ax_a.set_title("Legal reliability profile", loc="left")
    for i in range(len(METHODS)):
        for j in range(3):
            value = matrix[i, j]
            ax_a.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.0)
    ax_a.add_patch(Rectangle((-0.48, METHODS.index("Full framework") - 0.48), 2.96, 0.96, fill=False, ec=COLORS["Full framework"], lw=0.9))
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax_a, fraction=0.04, pad=0.02).ax.tick_params(labelsize=5.8)
    panel(ax_a, "a")

    x = summary["hallucination_rate"]["mean"]
    y = summary["citation_f1"]["mean"]
    for method in METHODS:
        ax_b.scatter(x.loc[method], y.loc[method], s=29 if method in HIGHLIGHTS else 18, color=COLORS[method], zorder=3)
        offset = (4, 3) if method not in {"Conflict-aware", "Full framework"} else (4, -9)
        ax_b.annotate(method, (x.loc[method], y.loc[method]), xytext=offset, textcoords="offset points", fontsize=5.8, color=COLORS["ink"])
    ax_b.set_xlabel("Hallucination rate (lower is better)")
    ax_b.set_ylabel("Citation F1 (higher is better)")
    ax_b.set_title("Reliability trade-off", loc="left")
    clean_axis(ax_b)
    panel(ax_b, "b")
    fig.subplots_adjust(top=0.90, bottom=0.21, left=0.15, right=0.97)
    save(fig, "fig_exp02_reliability")


def generate_dynamics(rng: np.random.Generator) -> pd.DataFrame:
    configurations = {
        "FedAvg": (0.660, 0.53, 0.035, 0.32, 23.0),
        "FedProx": (0.669, 0.53, 0.038, 0.30, 23.0),
        "SCAFFOLD": (0.679, 0.53, 0.042, 0.27, 25.5),
        "FedNova": (0.674, 0.53, 0.040, 0.28, 23.5),
        "FedLoRA": (0.690, 0.54, 0.042, 0.24, 4.1),
        "Conflict-aware": (0.709, 0.54, 0.044, 0.20, 4.5),
        "Full framework": (0.723, 0.54, 0.046, 0.18, 4.9),
    }
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for method, (ceiling, start, rate, drift0, payload) in configurations.items():
            for round_id in range(1, 101):
                progress = 1 - np.exp(-rate * round_id)
                accuracy = start + (ceiling - start) * progress + rng.normal(0, 0.004)
                drift = 0.052 + drift0 * np.exp(-0.022 * round_id) + rng.normal(0, 0.003)
                cumulative = payload * round_id + rng.normal(0, payload * 0.04)
                rows.append(
                    {
                        "seed": seed,
                        "round": round_id,
                        "method": method,
                        "legal_accuracy": accuracy,
                        "client_drift": max(drift, 0.01),
                        "cumulative_communication_mb": max(cumulative, 0),
                    }
                )
    return pd.DataFrame(rows)


def line_with_band(ax: plt.Axes, summary: pd.DataFrame, method: str, col: str) -> None:
    subset = summary[summary["method"] == method].sort_values("round")
    alpha = 1.0 if method in HIGHLIGHTS else 0.86
    lw = 1.65 if method in HIGHLIGHTS else 0.85
    ax.plot(
        subset["round"],
        subset["mean"],
        color=COLORS[method],
        lw=lw,
        linestyle=LINESTYLES[method],
        alpha=alpha,
        label=method,
    )
    if method in HIGHLIGHTS:
        ax.fill_between(
            subset["round"],
            subset["mean"] - subset["ci"],
            subset["mean"] + subset["ci"],
            color=COLORS[method],
            alpha=0.11,
            linewidth=0,
        )


def plot_dynamics(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45))
    panels = [
        ("legal_accuracy", "Validation accuracy", "Accuracy", (0.52, 0.74)),
        ("client_drift", "Client drift", "Update distance", (0.04, 0.38)),
        ("cumulative_communication_mb", "Communication", "Cumulative MB", None),
    ]
    for ax, (metric, title, ylabel, limits), label in zip(axes, panels, ["a", "b", "c"]):
        summary = mean_ci(data, ["round", "method"], metric)
        for method in FED_METHODS:
            line_with_band(ax, summary, method, metric)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 25, 50, 75, 100])
        if limits:
            ax.set_ylim(*limits)
        clean_axis(ax)
        panel(ax, label)
    draw_method_legend(fig, FED_METHODS, y=1.11)
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.075, right=0.98, wspace=0.41)
    save(fig, "fig_exp03_dynamics")


def generate_conflict(rng: np.random.Generator) -> pd.DataFrame:
    methods = ["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]
    component_base = {
        "citation": 0.58,
        "reasoning": 0.48,
        "verdict": 0.40,
        "rule alignment": 0.51,
    }
    factors = {"FedAvg": 1.0, "FedLoRA": 0.85, "Conflict-aware": 0.64, "Full framework": 0.55}
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for method in methods:
            for round_id in range(1, 101):
                scores: dict[str, float] = {}
                for component, start in component_base.items():
                    score = (
                        0.09
                        + start * factors[method] * np.exp(-0.014 * round_id)
                        + rng.normal(0, 0.007)
                    )
                    scores[component] = max(score, 0.0)
                    rows.append(
                        {
                            "seed": seed,
                            "round": round_id,
                            "method": method,
                            "component": component,
                            "score": scores[component],
                        }
                    )
                weighted = (
                    0.25 * scores["citation"]
                    + 0.30 * scores["reasoning"]
                    + 0.25 * scores["verdict"]
                    + 0.20 * scores["rule alignment"]
                )
                rows.append(
                    {
                        "seed": seed,
                        "round": round_id,
                        "method": method,
                        "component": "total",
                        "score": weighted,
                    }
                )
    return pd.DataFrame(rows)


def plot_conflict(data: pd.DataFrame) -> None:
    methods = ["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.65), gridspec_kw={"width_ratios": [1.22, 1]})
    total = data[data["component"] == "total"]
    summary = mean_ci(total, ["round", "method"], "score")
    for method in methods:
        line_with_band(axes[0], summary, method, "score")
    axes[0].set_title("Conflict over optimization", loc="left")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Weighted conflict score")
    axes[0].set_xticks([1, 25, 50, 75, 100])
    clean_axis(axes[0])
    panel(axes[0], "a")

    final = data[(data["round"] == 100) & (data["component"] != "total")]
    final_summary = mean_ci(final, ["component", "method"], "score")
    components = ["citation", "reasoning", "verdict", "rule alignment"]
    x = np.arange(len(components))
    for method in methods:
        subset = final_summary[final_summary["method"] == method].set_index("component").loc[components]
        axes[1].plot(
            x,
            subset["mean"],
            marker="o",
            ms=4.0 if method in HIGHLIGHTS else 3.1,
            lw=1.5 if method in HIGHLIGHTS else 0.9,
            linestyle=LINESTYLES.get(method, "-"),
            color=COLORS[method],
        )
    axes[1].set_xticks(x, ["Citation", "Reasoning", "Verdict", "Rule\nalignment"])
    axes[1].set_ylabel("Score at round 100")
    axes[1].set_title("Component diagnosis", loc="left")
    clean_axis(axes[1])
    panel(axes[1], "b")
    draw_method_legend(fig, methods, y=1.08)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.08, right=0.98, wspace=0.40)
    save(fig, "fig_exp04_conflict")


def generate_transfer(rng: np.random.Generator) -> pd.DataFrame:
    base = {
        "Local-only": 0.570,
        "Centralized": 0.616,
        "FedAvg": 0.606,
        "FedProx": 0.622,
        "FedLoRA": 0.645,
        "Conflict-aware": 0.676,
        "Full framework": 0.699,
    }
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for train in JURISDICTIONS:
            for test in JURISDICTIONS:
                same_system = train == test
                for method in CORE_TRANSFER_METHODS:
                    value = base[method] + (0.060 if same_system else -0.010) + rng.normal(0, 0.009)
                    rows.append(
                        {
                            "seed": seed,
                            "train_jurisdiction": train,
                            "test_jurisdiction": test,
                            "method": method,
                            "legal_accuracy": np.clip(value, 0.48, 0.84),
                        }
                    )
    return pd.DataFrame(rows)


def plot_transfer(data: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(WIDTH, 2.9))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.16], wspace=0.48)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    full = data[data["method"] == "Full framework"].groupby(
        ["train_jurisdiction", "test_jurisdiction"], sort=False
    )["legal_accuracy"].mean()
    matrix = (
        full.unstack("test_jurisdiction").loc[JURISDICTIONS, JURISDICTIONS].values
    )
    cmap = LinearSegmentedColormap.from_list("transfer", ["#F3F4F5", "#9FC0CA", "#175A8B"])
    im = ax_a.imshow(matrix, cmap=cmap, vmin=0.64, vmax=0.78)
    ax_a.set_xticks(np.arange(4), JURISDICTIONS, rotation=35, ha="right")
    ax_a.set_yticks(np.arange(4), JURISDICTIONS)
    ax_a.set_xlabel("Test jurisdiction")
    ax_a.set_ylabel("Train jurisdiction")
    ax_a.set_title("Full framework transfer", loc="left")
    for i in range(4):
        for j in range(4):
            ax_a.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.2)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax_a, fraction=0.047, pad=0.03).ax.tick_params(labelsize=5.8)
    panel(ax_a, "a")

    heldout = data[data["train_jurisdiction"] != data["test_jurisdiction"]]
    summary = mean_ci(heldout, ["method"], "legal_accuracy").set_index("method").loc[CORE_TRANSFER_METHODS]
    order = summary["mean"].sort_values().index.tolist()
    y = np.arange(len(order))
    for yi, method in zip(y, order):
        ax_b.errorbar(
            summary.loc[method, "mean"],
            yi,
            xerr=summary.loc[method, "ci"],
            fmt="o",
            color=COLORS[method],
            ms=4.1 if method in HIGHLIGHTS else 3.4,
            capsize=2,
            lw=1,
        )
    ax_b.axvline(summary.loc["FedAvg", "mean"], color=COLORS["grid"], lw=0.9)
    ax_b.set_yticks(y, order)
    ax_b.set_xlabel("Held-out jurisdiction accuracy")
    ax_b.set_title("Unseen jurisdiction performance", loc="left")
    ax_b.grid(axis="x", color=COLORS["grid"], lw=0.55)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    panel(ax_b, "b")
    fig.subplots_adjust(top=0.88, bottom=0.24, left=0.14, right=0.97)
    save(fig, "fig_exp05_transfer")


def generate_ablation(rng: np.random.Generator) -> pd.DataFrame:
    variants = {
        "Full framework": (0.000, 0.000, 0.000),
        "No jurisdiction embedding": (-0.024, -0.019, 0.010),
        "No citation conflict": (-0.017, -0.041, 0.028),
        "No reasoning conflict": (-0.020, -0.022, 0.020),
        "No verdict conflict": (-0.014, -0.010, 0.012),
        "No rule alignment": (-0.021, -0.026, 0.020),
        "No citation verifier": (-0.025, -0.050, 0.044),
        "No debate agents": (-0.018, -0.015, 0.018),
        "No privacy auditor": (-0.006, -0.005, 0.008),
    }
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for variant, (acc_delta, cit_delta, hall_delta) in variants.items():
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "legal_accuracy": 0.719 + acc_delta + rng.normal(0, 0.004),
                    "citation_f1": 0.764 + cit_delta + rng.normal(0, 0.005),
                    "hallucination_rate": 0.141 + hall_delta + rng.normal(0, 0.004),
                }
            )
    return pd.DataFrame(rows)


def plot_ablation(data: pd.DataFrame) -> None:
    full = data[data["variant"] == "Full framework"]
    variants = [item for item in data["variant"].drop_duplicates() if item != "Full framework"]
    rows = []
    for variant in variants:
        current = data[data["variant"] == variant].sort_values("seed")
        baseline = full.sort_values("seed")
        for metric in ["legal_accuracy", "citation_f1", "hallucination_rate"]:
            differences = current[metric].values - baseline[metric].values
            if metric == "hallucination_rate":
                differences *= -1
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": differences.mean() * 100,
                    "ci": 1.96 * differences.std(ddof=1) / np.sqrt(len(differences)) * 100,
                }
            )
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.12), sharey=True)
    specifications = [
        ("legal_accuracy", "Accuracy change"),
        ("citation_f1", "Citation F1 change"),
        ("hallucination_rate", "Hallucination reduction"),
    ]
    y = np.arange(len(variants))
    for ax, (metric, title), label in zip(axes, specifications, ["a", "b", "c"]):
        subset = summary[summary["metric"] == metric].set_index("variant").loc[variants]
        colors = [COLORS["placeholder"] if value < 0 else COLORS["Full framework"] for value in subset["mean"]]
        ax.axvline(0, color=COLORS["ink"], lw=0.65)
        for yi, value, ci, color in zip(y, subset["mean"], subset["ci"], colors):
            ax.errorbar(value, yi, xerr=ci, fmt="o", ms=3.8, color=color, lw=1.0, capsize=1.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Delta vs full (points)")
        ax.grid(axis="x", color=COLORS["grid"], lw=0.55)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel(ax, label)
    axes[0].set_yticks(y, variants)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    fig.subplots_adjust(top=0.90, bottom=0.20, left=0.27, right=0.98, wspace=0.40)
    save(fig, "fig_exp06_ablation")


def generate_privacy(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for secure in [False, True]:
            for noise in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5]:
                rows.append(
                    {
                        "seed": seed,
                        "noise_multiplier": noise,
                        "secure_aggregation": secure,
                        "legal_accuracy": 0.720 - 0.045 * noise + (0.001 if secure else 0) + rng.normal(0, 0.004),
                        "attack_auc": 0.79 - 0.17 * noise - (0.045 if secure else 0) + rng.normal(0, 0.008),
                        "communication_mb_per_round": 4.45 + (0.55 if secure else 0.0) + rng.normal(0, 0.05),
                    }
                )
    return pd.DataFrame(rows)


def plot_privacy(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.55))
    styles = [(False, "DP only", COLORS["FedLoRA"]), (True, "DP + secure agg.", COLORS["Full framework"])]
    specifications = [
        ("legal_accuracy", "Accuracy", "Legal accuracy"),
        ("attack_auc", "Membership inference", "Attack AUC"),
        ("communication_mb_per_round", "Payload overhead", "MB / round"),
    ]
    for ax, (metric, title, ylabel), label in zip(axes, specifications, ["a", "b", "c"]):
        summary = mean_ci(data, ["noise_multiplier", "secure_aggregation"], metric)
        for secure, legend, color in styles:
            sub = summary[summary["secure_aggregation"] == secure].sort_values("noise_multiplier")
            ax.errorbar(
                sub["noise_multiplier"],
                sub["mean"],
                yerr=sub["ci"],
                marker="o",
                ms=3.2,
                capsize=1.5,
                color=color,
                lw=1.35,
                label=legend,
            )
        ax.axvline(0.8, color=COLORS["grid"], lw=0.9, ls="--")
        ax.set_xlabel("DP noise multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        clean_axis(ax)
        panel(ax, label)
    axes[0].legend(loc="lower left", handlelength=1.7)
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.08, right=0.98, wspace=0.43)
    save(fig, "fig_exp07_privacy")


def generate_robustness(rng: np.random.Generator) -> pd.DataFrame:
    methods = ["FedAvg", "FedProx", "FedLoRA", "Conflict-aware", "Full framework"]
    gains = {"FedAvg": 0, "FedProx": 0.012, "FedLoRA": 0.030, "Conflict-aware": 0.050, "Full framework": 0.064}
    payload = {"FedAvg": 22.5, "FedProx": 22.7, "FedLoRA": 4.0, "Conflict-aware": 4.4, "Full framework": 4.8}
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for clients in [4, 8, 16, 32]:
            for alpha in [0.1, 0.3, 1.0]:
                penalty = {0.1: -0.045, 0.3: -0.020, 1.0: 0.0}[alpha]
                for method in methods:
                    rows.append(
                        {
                            "seed": seed,
                            "num_clients": clients,
                            "dirichlet_alpha": alpha,
                            "method": method,
                            "legal_accuracy": 0.651 + gains[method] + penalty - 0.005 * np.log2(clients / 4) + rng.normal(0, 0.005),
                            "total_communication_gb": payload[method] * clients * 100 / 1024,
                        }
                    )
    return pd.DataFrame(rows)


def plot_robustness(data: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(WIDTH, 2.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.05, 0.98], wspace=0.48)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    full = (
        data[data["method"] == "Full framework"]
        .groupby(["dirichlet_alpha", "num_clients"])["legal_accuracy"]
        .mean()
        .unstack("num_clients")
        .loc[[0.1, 0.3, 1.0], [4, 8, 16, 32]]
    )
    cmap = LinearSegmentedColormap.from_list("robust", ["#F3F4F5", "#90B4C2", "#175A8B"])
    im = axes[0].imshow(full.values, cmap=cmap, vmin=0.65, vmax=0.72, aspect="auto")
    axes[0].set_xticks(np.arange(4), [4, 8, 16, 32])
    axes[0].set_yticks(np.arange(3), [0.1, 0.3, 1.0])
    axes[0].set_xlabel("Clients")
    axes[0].set_ylabel("Dirichlet alpha")
    axes[0].set_title("Full framework", loc="left")
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, f"{full.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.9)
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    panel(axes[0], "a")

    subset = data[data["dirichlet_alpha"] == 0.3]
    methods = ["FedAvg", "FedProx", "FedLoRA", "Conflict-aware", "Full framework"]
    accuracy = mean_ci(subset, ["num_clients", "method"], "legal_accuracy")
    comm = subset.groupby(["num_clients", "method"], sort=False)["total_communication_gb"].mean().reset_index()
    for method in methods:
        acc = accuracy[accuracy["method"] == method].sort_values("num_clients")
        axes[1].plot(acc["num_clients"], acc["mean"], marker="o", ms=3.0, lw=1.4 if method in HIGHLIGHTS else 0.9, color=COLORS[method], linestyle=LINESTYLES.get(method, "-"))
        payload = comm[comm["method"] == method].sort_values("num_clients")
        axes[2].plot(payload["num_clients"], payload["total_communication_gb"], marker="o", ms=3.0, lw=1.4 if method in HIGHLIGHTS else 0.9, color=COLORS[method], linestyle=LINESTYLES.get(method, "-"))
    for ax, title, ylabel, label in [
        (axes[1], "Scale sensitivity", "Accuracy", "b"),
        (axes[2], "Communication cost", "Total GB / 100 rounds", "c"),
    ]:
        ax.set_xticks([4, 8, 16, 32])
        ax.set_xlabel("Clients")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        clean_axis(ax)
        panel(ax, label)
    draw_method_legend(fig, methods, y=1.08)
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.08, right=0.98)
    save(fig, "fig_exp08_robustness")


def generate_all(use_existing: bool, seed: int, final_data: bool) -> None:
    global PLACEHOLDER_MODE
    PLACEHOLDER_MODE = not final_data
    ensure_dirs()
    configure_style()
    rng = np.random.default_rng(seed)

    benchmark = load_or_generate("benchmark_results.csv", generate_benchmark, rng, use_existing)
    dynamics = load_or_generate("training_dynamics.csv", generate_dynamics, rng, use_existing)
    conflict = load_or_generate("conflict_diagnostics.csv", generate_conflict, rng, use_existing)
    transfer = load_or_generate("cross_jurisdiction.csv", generate_transfer, rng, use_existing)
    ablation = load_or_generate("ablation.csv", generate_ablation, rng, use_existing)
    privacy = load_or_generate("privacy_utility.csv", generate_privacy, rng, use_existing)
    robustness = load_or_generate("robustness.csv", generate_robustness, rng, use_existing)

    plot_benchmark(benchmark)
    plot_reliability(benchmark)
    plot_dynamics(dynamics)
    plot_conflict(conflict)
    plot_transfer(transfer)
    plot_ablation(ablation)
    plot_privacy(privacy)
    plot_robustness(robustness)
    print(f"Source CSV templates: {DATA_DIR}")
    print(f"Publication figure templates: {FIG_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-existing-data",
        action="store_true",
        help="Use the CSV files already present in the top-conference source-data directory.",
    )
    parser.add_argument(
        "--final-data",
        action="store_true",
        help="Hide the synthetic-data footer only after verified result values have been inserted.",
    )
    parser.add_argument("--seed", type=int, default=19, help="Random seed for synthetic template creation.")
    args = parser.parse_args()
    generate_all(args.use_existing_data, args.seed, args.final_data)


if __name__ == "__main__":
    main()
