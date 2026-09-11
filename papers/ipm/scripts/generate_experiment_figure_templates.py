"""Generate conference-style experiment figure templates for the IPM paper.

The script creates random but plausible placeholder data with a fixed seed and
writes both source CSV files and publication-ready figures. Replace the CSV
values with real experiment results and rerun with ``--use-existing-data`` to
redraw the figures without regenerating random values. Use ``--final-data``
only after all displayed values have been verified as real run results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = PAPER_DIR / "figures" / "experiments"
DATA_DIR = PAPER_DIR / "tables" / "experiment_source_data"

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

FED_METHODS = ["FedAvg", "FedProx", "SCAFFOLD", "FedNova", "FedLoRA", "Conflict-aware", "Full framework"]
JURISDICTIONS = ["Chinese civil law", "US common law", "EU regulatory", "Contract law"]
DATASETS = ["CAIL", "CaseHOLD", "ECtHR", "CUAD"]

PALETTE = {
    "Centralized": "#6B7280",
    "Local-only": "#B8C1CC",
    "FedAvg": "#9AA7B8",
    "FedProx": "#7D8FA6",
    "SCAFFOLD": "#6C9BC8",
    "FedNova": "#8CB6D8",
    "FedLoRA": "#8E75B6",
    "Conflict-aware": "#C9805C",
    "Full framework": "#2F6F9F",
    "accent": "#2F6F9F",
    "accent2": "#8E75B6",
    "green": "#4F9D69",
    "red": "#C75D5D",
    "gold": "#C49A3A",
    "grid": "#E8EDF3",
    "ink": "#27313F",
    "muted": "#697386",
}

PLACEHOLDER_MODE = True


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def save(fig: plt.Figure, name: str) -> None:
    if PLACEHOLDER_MODE:
        fig.text(
            0.995,
            0.002,
            "SYNTHETIC PLACEHOLDER DATA - REPLACE BEFORE SUBMISSION",
            ha="right",
            va="bottom",
            fontsize=6.0,
            color=PALETTE["red"],
            fontweight="bold",
        )
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def read_or_create(path: Path, create_fn, use_existing: bool) -> pd.DataFrame:
    if use_existing and path.exists():
        return pd.read_csv(path)
    df = create_fn()
    df.to_csv(path, index=False)
    return df


def add_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.set_axisbelow(True)


def label_panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def make_main_performance(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    base = {
        "Centralized": 0.675,
        "Local-only": 0.600,
        "FedAvg": 0.635,
        "FedProx": 0.650,
        "SCAFFOLD": 0.662,
        "FedNova": 0.657,
        "FedLoRA": 0.681,
        "Conflict-aware": 0.705,
        "Full framework": 0.728,
    }
    for dataset_i, dataset in enumerate(DATASETS):
        shift = [-0.015, 0.005, -0.025, 0.015][dataset_i]
        for method in METHODS:
            mean = np.clip(base[method] + shift + rng.normal(0, 0.006), 0.50, 0.82)
            stderr = rng.uniform(0.006, 0.014)
            rows.append({"dataset": dataset, "method": method, "legal_accuracy": mean, "stderr": stderr})
    return pd.DataFrame(rows)


def plot_main_performance(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    axes = axes.ravel()
    x = np.arange(len(METHODS))
    for ax, dataset in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset].set_index("method").loc[METHODS].reset_index()
        colors = [PALETTE[m] for m in METHODS]
        ax.bar(x, sub["legal_accuracy"], yerr=sub["stderr"], color=colors, edgecolor="white", linewidth=0.4, capsize=2)
        ax.set_title(dataset, loc="left")
        ax.set_xticks(x, METHODS, rotation=38, ha="right")
        ax.set_ylim(0.52, 0.79)
        add_grid(ax)
    axes[0].set_ylabel("legal accuracy")
    axes[2].set_ylabel("legal accuracy")
    label_panel(axes[0], "a")
    fig.suptitle("Main legal task performance", x=0.02, y=1.02, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    save(fig, "exp1_main_performance")


def make_reliability(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        level = METHODS.index(method) / (len(METHODS) - 1)
        rows.append(
            {
                "method": method,
                "citation_consistency": np.clip(0.55 + 0.23 * level + rng.normal(0, 0.01), 0, 1),
                "reasoning_coherence": np.clip(0.58 + 0.20 * level + rng.normal(0, 0.012), 0, 1),
                "hallucination_rate": np.clip(0.30 - 0.15 * level + rng.normal(0, 0.008), 0, 1),
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True)
    metrics = [
        ("citation_consistency", "citation consistency", PALETTE["green"], True),
        ("reasoning_coherence", "reasoning coherence", PALETTE["gold"], True),
        ("hallucination_rate", "hallucination rate", PALETTE["red"], False),
    ]
    x = np.arange(len(METHODS))
    for ax, (col, title, color, higher_better) in zip(axes, metrics):
        sub = df.set_index("method").loc[METHODS].reset_index()
        ax.bar(x, sub[col], color=[PALETTE[m] if m in ["Conflict-aware", "Full framework"] else color for m in METHODS], alpha=0.9, edgecolor="white", linewidth=0.4)
        ax.set_title(title + (" ↑" if higher_better else " ↓"), loc="left")
        ax.set_xticks(x, METHODS, rotation=42, ha="right")
        ax.set_ylim(0, 1.0)
        add_grid(ax)
    axes[0].set_ylabel("score")
    label_panel(axes[0], "a")
    fig.tight_layout()
    save(fig, "exp2_legal_reliability")


def make_training_dynamics(rng: np.random.Generator) -> pd.DataFrame:
    rounds = np.arange(1, 101)
    rows = []
    for method, asymptote, start, rate in [
        ("FedAvg", 0.665, 0.52, 0.035),
        ("FedProx", 0.675, 0.52, 0.038),
        ("SCAFFOLD", 0.690, 0.52, 0.042),
        ("FedLoRA", 0.700, 0.54, 0.040),
        ("Conflict-aware", 0.718, 0.54, 0.043),
        ("Full framework", 0.738, 0.55, 0.046),
    ]:
        curve = asymptote - (asymptote - start) * np.exp(-rate * rounds)
        curve += rng.normal(0, 0.003, size=len(rounds))
        drift = (0.35 - 0.10 * (METHODS.index(method) / len(METHODS))) * np.exp(-0.018 * rounds)
        drift += 0.05 + rng.normal(0, 0.002, size=len(rounds))
        for r, acc, d in zip(rounds, curve, drift):
            rows.append({"round": r, "method": method, "legal_accuracy": acc, "client_drift": max(d, 0.01)})
    return pd.DataFrame(rows)


def plot_training_dynamics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for method in ["FedAvg", "FedProx", "SCAFFOLD", "FedLoRA", "Conflict-aware", "Full framework"]:
        sub = df[df["method"] == method]
        axes[0].plot(sub["round"], sub["legal_accuracy"], label=method, color=PALETTE.get(method, "#777777"), lw=1.5)
        axes[1].plot(sub["round"], sub["client_drift"], label=method, color=PALETTE.get(method, "#777777"), lw=1.5)
    axes[0].set_title("convergence", loc="left")
    axes[0].set_xlabel("federated round")
    axes[0].set_ylabel("legal accuracy")
    axes[1].set_title("client drift", loc="left")
    axes[1].set_xlabel("federated round")
    axes[1].set_ylabel("drift")
    for ax in axes:
        add_grid(ax)
    axes[0].legend(ncol=2, fontsize=6.1)
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "exp3_training_dynamics")


def make_conflict_diagnostics(rng: np.random.Generator) -> pd.DataFrame:
    rounds = np.arange(1, 101)
    rows = []
    for method, factor in [("FedAvg", 1.0), ("FedLoRA", 0.82), ("Conflict-aware", 0.62), ("Full framework", 0.50)]:
        citation = 0.62 * factor * np.exp(-0.010 * rounds) + 0.12 + rng.normal(0, 0.006, len(rounds))
        reasoning = 0.45 * factor * np.exp(-0.012 * rounds) + 0.08 + rng.normal(0, 0.006, len(rounds))
        verdict = 0.38 * factor * np.exp(-0.009 * rounds) + 0.09 + rng.normal(0, 0.005, len(rounds))
        rule = 0.52 * factor * np.exp(-0.008 * rounds) + 0.10 + rng.normal(0, 0.005, len(rounds))
        total = 0.25 * citation + 0.30 * reasoning + 0.25 * verdict + 0.20 * rule
        for i, r in enumerate(rounds):
            rows.append(
                {
                    "round": r,
                    "method": method,
                    "citation_conflict": max(citation[i], 0),
                    "reasoning_conflict": max(reasoning[i], 0),
                    "verdict_conflict": max(verdict[i], 0),
                    "rule_alignment_distance": max(rule[i], 0),
                    "total_conflict": max(total[i], 0),
                }
            )
    return pd.DataFrame(rows)


def plot_conflict_diagnostics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for method in ["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]:
        sub = df[df["method"] == method]
        axes[0].plot(sub["round"], sub["total_conflict"], label=method, color=PALETTE.get(method, "#777777"), lw=1.5)
    final = df[df["round"] == df["round"].max()].set_index("method").loc[["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]]
    components = ["citation_conflict", "reasoning_conflict", "verdict_conflict", "rule_alignment_distance"]
    bottom = np.zeros(len(final))
    colors = ["#6C9BC8", "#C9805C", "#C75D5D", "#8E75B6"]
    x = np.arange(len(final))
    for comp, color in zip(components, colors):
        axes[1].bar(x, final[comp], bottom=bottom, label=comp.replace("_", " "), color=color, edgecolor="white", linewidth=0.4)
        bottom += final[comp].values
    axes[0].set_title("total conflict over rounds", loc="left")
    axes[0].set_xlabel("federated round")
    axes[0].set_ylabel("total conflict")
    axes[1].set_title("final conflict composition", loc="left")
    axes[1].set_xticks(x, final.index, rotation=25, ha="right")
    axes[1].set_ylabel("component sum")
    axes[0].legend(fontsize=6.2)
    axes[1].legend(fontsize=5.8)
    for ax in axes:
        add_grid(ax)
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "exp4_conflict_diagnostics")


def make_cross_jurisdiction(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    methods = ["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]
    base = {"FedAvg": 0.60, "FedLoRA": 0.64, "Conflict-aware": 0.68, "Full framework": 0.71}
    for train in JURISDICTIONS:
        for test in JURISDICTIONS:
            for method in methods:
                same = train == test
                value = base[method] + (0.06 if same else -0.03) + rng.normal(0, 0.009)
                rows.append({"train_jurisdiction": train, "test_jurisdiction": test, "method": method, "accuracy": np.clip(value, 0.45, 0.82)})
    return pd.DataFrame(rows)


def plot_cross_jurisdiction(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    full = df[df["method"] == "Full framework"]
    pivot = full.pivot(index="train_jurisdiction", columns="test_jurisdiction", values="accuracy").loc[JURISDICTIONS, JURISDICTIONS]
    im = axes[0].imshow(pivot.values, cmap="Blues", vmin=0.50, vmax=0.80)
    axes[0].set_xticks(np.arange(len(JURISDICTIONS)), JURISDICTIONS, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(JURISDICTIONS)), JURISDICTIONS)
    axes[0].set_title("Full framework transfer matrix", loc="left")
    for i in range(len(JURISDICTIONS)):
        for j in range(len(JURISDICTIONS)):
            axes[0].text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=6.2, color=PALETTE["ink"])
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    holdout = df[df["train_jurisdiction"] != df["test_jurisdiction"]].groupby("method", as_index=False)["accuracy"].mean()
    holdout = holdout.set_index("method").loc[["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]].reset_index()
    axes[1].bar(np.arange(len(holdout)), holdout["accuracy"], color=[PALETTE[m] for m in holdout["method"]], edgecolor="white", linewidth=0.4)
    axes[1].set_xticks(np.arange(len(holdout)), holdout["method"], rotation=30, ha="right")
    axes[1].set_ylim(0.50, 0.75)
    axes[1].set_ylabel("held-out jurisdiction accuracy")
    axes[1].set_title("Cross-jurisdiction generalization", loc="left")
    add_grid(axes[1])
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "exp5_cross_jurisdiction")


def make_ablation(rng: np.random.Generator) -> pd.DataFrame:
    variants = [
        "Full framework",
        "no jurisdiction embedding",
        "no citation penalty",
        "no reasoning penalty",
        "no verdict penalty",
        "no rule alignment",
        "no citation verifier",
        "no debate agents",
        "no privacy auditor",
    ]
    rows = []
    for i, variant in enumerate(variants):
        drop = [0.0, 0.025, 0.018, 0.022, 0.016, 0.020, 0.028, 0.021, 0.010][i]
        rows.append(
            {
                "variant": variant,
                "legal_accuracy": 0.728 - drop + rng.normal(0, 0.004),
                "citation_consistency": 0.780 - drop * 1.4 + rng.normal(0, 0.005),
                "hallucination_rate": 0.142 + drop * 1.1 + rng.normal(0, 0.004),
                "communication_cost": 0.22 + (0.00 if i == 0 else rng.normal(0.01, 0.006)),
            }
        )
    return pd.DataFrame(rows)


def plot_ablation(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ordered = df.sort_values("legal_accuracy")
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["legal_accuracy"], color=[PALETTE["accent"] if v == "Full framework" else "#9AA7B8" for v in ordered["variant"]])
    axes[0].set_yticks(y, ordered["variant"])
    axes[0].set_xlabel("legal accuracy")
    axes[0].set_xlim(0.66, 0.75)
    axes[0].set_title("module removal impact", loc="left")
    add_grid(axes[0])
    metrics = ["legal_accuracy", "citation_consistency", "hallucination_rate", "communication_cost"]
    normalized = df.set_index("variant")[metrics]
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    im = axes[1].imshow(normalized.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axes[1].set_yticks(np.arange(len(normalized.index)), normalized.index)
    axes[1].set_xticks(np.arange(len(metrics)), [m.replace("_", "\n") for m in metrics])
    axes[1].set_title("normalized ablation profile", loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "exp6_ablation")


def make_privacy_utility(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for noise in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5]:
        privacy = 1 - np.exp(-1.5 * noise)
        for secure in [False, True]:
            utility = 0.73 - 0.055 * noise + (0.005 if secure else 0.0) + rng.normal(0, 0.004)
            leakage = 0.42 * np.exp(-1.4 * noise) * (0.75 if secure else 1.0) + rng.normal(0, 0.006)
            rows.append(
                {
                    "noise_multiplier": noise,
                    "secure_aggregation": secure,
                    "legal_accuracy": np.clip(utility, 0.55, 0.75),
                    "privacy_strength": privacy,
                    "leakage_risk": np.clip(leakage, 0.02, 0.45),
                }
            )
    return pd.DataFrame(rows)


def plot_privacy_utility(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for secure, label, color in [(False, "DP only", PALETTE["accent2"] if "accent2" in PALETTE else "#8E75B6"), (True, "DP + secure aggregation", PALETTE["accent"])]:
        sub = df[df["secure_aggregation"] == secure]
        axes[0].plot(sub["noise_multiplier"], sub["legal_accuracy"], marker="o", label=label, color=color)
        axes[1].plot(sub["noise_multiplier"], sub["leakage_risk"], marker="o", label=label, color=color)
    axes[0].set_title("privacy-utility trade-off", loc="left")
    axes[0].set_xlabel("DP noise multiplier")
    axes[0].set_ylabel("legal accuracy")
    axes[1].set_title("leakage-risk reduction", loc="left")
    axes[1].set_xlabel("DP noise multiplier")
    axes[1].set_ylabel("leakage risk")
    for ax in axes:
        add_grid(ax)
        ax.legend()
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "exp7_privacy_utility")


def make_scaling_sensitivity(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for clients in [4, 8, 16, 32]:
        for alpha in [0.1, 0.3, 1.0]:
            noniid_penalty = {0.1: 0.050, 0.3: 0.025, 1.0: 0.000}[alpha]
            for method in ["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]:
                method_gain = {"FedAvg": 0.00, "FedLoRA": 0.025, "Conflict-aware": 0.045, "Full framework": 0.060}[method]
                value = 0.66 + method_gain - noniid_penalty - 0.006 * np.log2(clients / 4) + rng.normal(0, 0.004)
                comm = (1.0 if method == "FedAvg" else 0.32 if method == "FedLoRA" else 0.36 if method == "Conflict-aware" else 0.40) * clients
                rows.append({"num_clients": clients, "dirichlet_alpha": alpha, "method": method, "accuracy": value, "relative_communication": comm})
    return pd.DataFrame(rows)


def plot_scaling_sensitivity(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for method in ["FedAvg", "FedLoRA", "Conflict-aware", "Full framework"]:
        sub = df[(df["dirichlet_alpha"] == 0.3) & (df["method"] == method)]
        axes[0].plot(sub["num_clients"], sub["accuracy"], marker="o", label=method, color=PALETTE.get(method, "#777777"))
    pivot = df[df["method"] == "Full framework"].pivot(index="dirichlet_alpha", columns="num_clients", values="accuracy")
    im = axes[1].imshow(pivot.values, cmap="YlGnBu", vmin=df["accuracy"].min(), vmax=df["accuracy"].max(), aspect="auto")
    axes[0].set_title("client scaling", loc="left")
    axes[0].set_xlabel("number of clients")
    axes[0].set_ylabel("accuracy")
    axes[0].set_xticks([4, 8, 16, 32])
    axes[0].legend(fontsize=6.0)
    axes[1].set_title("non-IID sensitivity, full framework", loc="left")
    axes[1].set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    axes[1].set_yticks(np.arange(len(pivot.index)), pivot.index)
    axes[1].set_xlabel("number of clients")
    axes[1].set_ylabel("Dirichlet alpha")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    for ax in [axes[0]]:
        add_grid(ax)
    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "exp8_scaling_sensitivity")


def generate_all(use_existing: bool, seed: int, final_data: bool) -> None:
    global PLACEHOLDER_MODE
    PLACEHOLDER_MODE = not final_data
    ensure_dirs()
    setup_style()
    rng = np.random.default_rng(seed)
    specs = [
        ("main_performance.csv", make_main_performance, plot_main_performance),
        ("legal_reliability.csv", make_reliability, plot_reliability),
        ("training_dynamics.csv", make_training_dynamics, plot_training_dynamics),
        ("conflict_diagnostics.csv", make_conflict_diagnostics, plot_conflict_diagnostics),
        ("cross_jurisdiction.csv", make_cross_jurisdiction, plot_cross_jurisdiction),
        ("ablation.csv", make_ablation, plot_ablation),
        ("privacy_utility.csv", make_privacy_utility, plot_privacy_utility),
        ("scaling_sensitivity.csv", make_scaling_sensitivity, plot_scaling_sensitivity),
    ]
    for filename, make_fn, plot_fn in specs:
        df = read_or_create(DATA_DIR / filename, lambda fn=make_fn: fn(rng), use_existing)
        plot_fn(df)
    print(f"Wrote source CSVs to {DATA_DIR}")
    print(f"Wrote experiment figures to {FIG_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-existing-data", action="store_true", help="Read existing CSV files instead of regenerating random placeholders.")
    parser.add_argument("--final-data", action="store_true", help="Remove the placeholder footer only after the CSV files have been replaced by verified experimental results.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate_all(args.use_existing_data, args.seed, args.final_data)


if __name__ == "__main__":
    main()
