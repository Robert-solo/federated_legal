"""Generate ACL-style figures from the latest second-round real experiment.

This script targets the richer real run that emits three-round server
evaluation, client evaluation, client drift, train loss, and communication logs.
It renders only metrics present in the run artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "real_flower_peft"
FIG_DIR = PAPER_DIR / "figures" / "experiments_acl_round2"
DATA_DIR = PAPER_DIR / "tables" / "experiment_source_data_acl_round2"

WIDTH_DOUBLE = 7.12
WIDTH_SINGLE = 3.42

INK = "#202124"
MUTED = "#5F6772"
GRID = "#E6E8EB"
LIGHT = "#F5F6F7"
BLUE = "#2F6C99"
TEAL = "#2A8C82"
ORANGE = "#C87533"
RED = "#B44B4B"
GRAY = "#9AA2AA"
PALE_BLUE = "#E8F0F5"


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
            "axes.titlesize": 7.8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.7,
            "axes.edgecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def latest_eval_run(run_root: Path) -> Path:
    candidates = sorted(
        [
            p
            for p in run_root.glob("*")
            if (p / "real_experiment_report.json").exists()
            and (p / "server_eval_rounds.jsonl").exists()
            and (p / "client_eval_log.jsonl").exists()
        ],
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No real eval run found under {run_root}")
    return candidates[-1]


def records_from_comm(comm_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in comm_rows:
        metrics = row.get("metrics", {})
        rows.append(
            {
                "client_id": row.get("client_id", "client"),
                "round": int(float(metrics.get("server_round", 0))),
                "train_loss": float(metrics.get("train_loss", np.nan)),
                "client_drift_l2": float(metrics.get("client_drift_l2", np.nan)),
                "client_drift_ratio": float(metrics.get("client_drift_ratio", np.nan)),
                "elapsed_sec_train": float(metrics.get("elapsed_sec", np.nan)),
                "num_examples": int(float(metrics.get("num_examples", 0))),
                "uplink_bytes": int(float(row.get("uplink_bytes", metrics.get("uplink_bytes", 0)))),
                "downlink_bytes": int(float(row.get("downlink_bytes", metrics.get("downlink_bytes", 0)))),
                "total_bytes": int(float(row.get("num_bytes", metrics.get("total_bytes", 0)))),
                "timestamp": float(row.get("timestamp", np.nan)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("communication_log.jsonl contains no rows")
    frame = frame.sort_values(["round", "client_id"]).reset_index(drop=True)
    frame["total_mib"] = frame["total_bytes"] / 1024**2
    frame["uplink_mib"] = frame["uplink_bytes"] / 1024**2
    frame["downlink_mib"] = frame["downlink_bytes"] / 1024**2
    return frame


def records_from_server_eval(rows: list[dict[str, Any]]) -> pd.DataFrame:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        parsed.append(
            {
                "round": int(row.get("round", 0)),
                "accuracy": float(metrics.get("accuracy", np.nan)),
                "correct": int(float(metrics.get("correct", 0))),
                "eval_loss": float(metrics.get("eval_loss", row.get("eval_loss", np.nan))),
                "num_examples": int(float(metrics.get("num_examples", 0))),
                "num_results": int(row.get("num_results", 0)),
                "num_failures": int(row.get("num_failures", 0)),
                "timestamp": float(row.get("timestamp", np.nan)),
            }
        )
    frame = pd.DataFrame(parsed)
    if frame.empty:
        raise ValueError("server_eval_rounds.jsonl contains no rows")
    return frame.sort_values("round").reset_index(drop=True)


def records_from_client_eval(rows: list[dict[str, Any]]) -> pd.DataFrame:
    parsed: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics", {})
        parsed.append(
            {
                "client_id": row.get("client_id", "client"),
                "round": int(float(metrics.get("server_round", 0))),
                "accuracy": float(metrics.get("accuracy", np.nan)),
                "correct": int(float(metrics.get("correct", 0))),
                "eval_loss": float(metrics.get("eval_loss", np.nan)),
                "num_examples": int(float(metrics.get("num_examples", 0))),
                "downlink_bytes_eval": int(float(metrics.get("downlink_bytes", 0))),
                "elapsed_sec_eval": float(metrics.get("elapsed_sec", np.nan)),
                "timestamp": float(row.get("timestamp", np.nan)),
            }
        )
    frame = pd.DataFrame(parsed)
    if frame.empty:
        raise ValueError("client_eval_log.jsonl contains no rows")
    frame = frame.sort_values(["round", "client_id"]).reset_index(drop=True)
    frame["downlink_mib_eval"] = frame["downlink_bytes_eval"] / 1024**2
    return frame


def records_from_manifest(manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    partition = manifest.get("partition_summary", {})
    size_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    jurisdiction_rows: list[dict[str, Any]] = []
    for client_id, summary in partition.items():
        size_rows.append(
            {
                "client_id": client_id,
                "num_examples": int(summary.get("num_examples", 0)),
                "num_jurisdictions": len(summary.get("jurisdictions", {})),
            }
        )
        for label, count in summary.get("labels", {}).items():
            label_rows.append({"client_id": client_id, "label": str(label), "count": int(count)})
        jurisdictions = summary.get("jurisdictions", {})
        total = max(sum(int(v) for v in jurisdictions.values()), 1)
        top_name, top_count = "", 0
        if jurisdictions:
            top_name, top_count = max(jurisdictions.items(), key=lambda item: int(item[1]))
        jurisdiction_rows.append(
            {
                "client_id": client_id,
                "top_jurisdiction": top_name,
                "top_jurisdiction_count": int(top_count),
                "top_jurisdiction_share": int(top_count) / total,
                "num_jurisdictions": len(jurisdictions),
            }
        )
    return pd.DataFrame(size_rows), pd.DataFrame(label_rows), pd.DataFrame(jurisdiction_rows)


def load_run(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = read_json(run_dir / "real_experiment_report.json")
    manifest = read_json(run_dir / "manifest.json")
    comm_df = records_from_comm(read_jsonl(run_dir / "communication_log.jsonl"))
    server_eval_df = records_from_server_eval(read_jsonl(run_dir / "server_eval_rounds.jsonl"))
    client_eval_df = records_from_client_eval(read_jsonl(run_dir / "client_eval_log.jsonl"))
    size_df, label_df, jurisdiction_df = records_from_manifest(manifest)

    profile = {
        "run_name": report.get("run_name", run_dir.name),
        "dataset": report.get("dataset", f"{manifest.get('dataset')}/{manifest.get('dataset_config')}"),
        "model_name": report.get("model_name", manifest.get("model_name", "")),
        "partition_strategy": manifest.get("partition_strategy", ""),
        "dirichlet_alpha": manifest.get("dirichlet_alpha", np.nan),
        "num_clients": int(report.get("num_clients", manifest.get("num_clients", 0))),
        "num_rounds": int(report.get("num_rounds", report.get("server_rounds", 0))),
        "eval_rounds": int(report.get("eval_rounds", len(server_eval_df))),
        "train_records": count_jsonl(run_dir / "train.jsonl"),
        "eval_records": count_jsonl(run_dir / "eval.jsonl"),
        "final_accuracy": float(report.get("final_accuracy", server_eval_df["accuracy"].iloc[-1])),
        "best_accuracy": float(report.get("best_accuracy", server_eval_df["accuracy"].max())),
        "first_accuracy": float(server_eval_df["accuracy"].iloc[0]),
        "accuracy_gain": float(server_eval_df["accuracy"].iloc[-1] - server_eval_df["accuracy"].iloc[0]),
        "final_eval_loss": float(server_eval_df["eval_loss"].iloc[-1]),
        "first_eval_loss": float(server_eval_df["eval_loss"].iloc[0]),
        "eval_loss_drop": float(server_eval_df["eval_loss"].iloc[0] - server_eval_df["eval_loss"].iloc[-1]),
        "mean_train_loss": float(report.get("mean_train_loss", comm_df["train_loss"].mean())),
        "mean_client_drift_l2": float(report.get("mean_client_drift_l2", comm_df["client_drift_l2"].mean())),
        "communication_bytes": int(report.get("communication_bytes", comm_df["total_bytes"].sum())),
        "communication_mib": float(report.get("communication_bytes", comm_df["total_bytes"].sum())) / 1024**2,
        "uplink_mib": float(report.get("uplink_bytes", comm_df["uplink_bytes"].sum())) / 1024**2,
        "downlink_mib": float(report.get("downlink_bytes", comm_df["downlink_bytes"].sum())) / 1024**2,
        "communication_events": int(report.get("communication_events", len(comm_df))),
        "run_dir": str(run_dir),
    }
    profile_df = pd.DataFrame([profile])
    return comm_df, server_eval_df, client_eval_df, size_df, label_df, jurisdiction_df, profile_df


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_source_data(
    comm_df: pd.DataFrame,
    server_eval_df: pd.DataFrame,
    client_eval_df: pd.DataFrame,
    size_df: pd.DataFrame,
    label_df: pd.DataFrame,
    jurisdiction_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    comm_df.to_csv(DATA_DIR / "round2_client_training_communication.csv", index=False)
    server_eval_df.to_csv(DATA_DIR / "round2_server_eval_rounds.csv", index=False)
    client_eval_df.to_csv(DATA_DIR / "round2_client_eval_log.csv", index=False)
    size_df.to_csv(DATA_DIR / "round2_partition_sizes.csv", index=False)
    label_df.to_csv(DATA_DIR / "round2_label_distribution.csv", index=False)
    jurisdiction_df.to_csv(DATA_DIR / "round2_jurisdiction_summary.csv", index=False)
    profile_df.to_csv(DATA_DIR / "round2_run_profile.csv", index=False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )


def clean_axis(ax: plt.Axes, *, grid: bool = True, axis: str = "y") -> None:
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid:
        ax.grid(axis=axis, color=GRID, lw=0.55, zorder=0)
    ax.set_axisbelow(True)


def annotate_status(fig: plt.Figure, profile_df: pd.DataFrame) -> None:
    p = profile_df.iloc[0]
    note = (
        f"REAL ROUND-2 RUN: {p['dataset']} | {int(p['num_clients'])} clients | "
        f"{int(p['num_rounds'])} rounds | {int(p['eval_records'])} eval cases | "
        "citation/hallucination metrics not emitted by this run"
    )
    fig.text(0.995, 0.006, note, ha="right", va="bottom", fontsize=5.5, color=MUTED)


def save_fig(fig: plt.Figure, name: str, profile_df: pd.DataFrame) -> None:
    annotate_status(fig, profile_df)
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 450}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def draw_partition_profile(
    size_df: pd.DataFrame,
    label_df: pd.DataFrame,
    jurisdiction_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    p = profile_df.iloc[0]
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 4.05))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.0], hspace=0.62, wspace=0.48)

    ax_sizes = fig.add_subplot(gs[0, 0])
    size_df = size_df.sort_values("client_id")
    x = np.arange(len(size_df))
    ax_sizes.bar(x, size_df["num_examples"], color=BLUE, width=0.58, edgecolor="white", linewidth=0.6)
    for xi, value in zip(x, size_df["num_examples"]):
        ax_sizes.text(xi, value + 12, f"{int(value)}", ha="center", va="bottom", fontsize=6.0)
    ax_sizes.set_xticks(x)
    ax_sizes.set_xticklabels(size_df["client_id"])
    ax_sizes.set_ylabel("train examples")
    ax_sizes.set_title("Non-IID client size")
    ax_sizes.set_ylim(0, size_df["num_examples"].max() * 1.20)
    clean_axis(ax_sizes)
    panel_label(ax_sizes, "a")

    ax_heat = fig.add_subplot(gs[0, 1])
    label_mat = (
        label_df.pivot_table(index="client_id", columns="label", values="count", fill_value=0)
        .sort_index()
        .sort_index(axis=1)
    )
    row_pct = label_mat.div(label_mat.sum(axis=1), axis=0)
    cmap = LinearSegmentedColormap.from_list("acl_blues", ["#F4F7FA", "#B7CCD9", BLUE])
    im = ax_heat.imshow(row_pct.values, aspect="auto", cmap=cmap, vmin=0, vmax=row_pct.values.max())
    ax_heat.set_xticks(np.arange(row_pct.shape[1]))
    ax_heat.set_xticklabels([str(c) for c in row_pct.columns])
    ax_heat.set_yticks(np.arange(row_pct.shape[0]))
    ax_heat.set_yticklabels(row_pct.index)
    ax_heat.set_xlabel("CaseHOLD label")
    ax_heat.set_title("Label composition")
    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            ax_heat.text(j, i, f"{row_pct.iat[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.6)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    cbar.set_label("% within client", fontsize=6.0)
    panel_label(ax_heat, "b")

    ax_juris = fig.add_subplot(gs[0, 2])
    jurisdiction_df = jurisdiction_df.sort_values("client_id")
    ax_juris.barh(
        np.arange(len(jurisdiction_df)),
        jurisdiction_df["top_jurisdiction_share"] * 100,
        color=ORANGE,
        height=0.46,
    )
    ax_juris.set_yticks(np.arange(len(jurisdiction_df)))
    ax_juris.set_yticklabels(jurisdiction_df["client_id"])
    ax_juris.invert_yaxis()
    ax_juris.set_xlabel("top jurisdiction share (%)")
    ax_juris.set_title("Jurisdiction concentration")
    ax_juris.set_xlim(0, max(60, jurisdiction_df["top_jurisdiction_share"].max() * 118))
    for yi, row in jurisdiction_df.reset_index(drop=True).iterrows():
        ax_juris.text(
            row["top_jurisdiction_share"] * 100 + 1.0,
            yi,
            f"{row['top_jurisdiction_share'] * 100:.0f}",
            va="center",
            ha="left",
            fontsize=6.0,
        )
    clean_axis(ax_juris, axis="x")
    panel_label(ax_juris, "c")

    ax_cards = fig.add_subplot(gs[1, :])
    ax_cards.axis("off")
    metrics = [
        ("model", str(p["model_name"]).replace("Qwen/", "")),
        ("partition", f"{p['partition_strategy']}, alpha={p['dirichlet_alpha']}"),
        ("train/eval", f"{int(p['train_records'])}/{int(p['eval_records'])}"),
        ("rounds", f"{int(p['num_rounds'])} train + {int(p['eval_rounds'])} eval"),
        ("final accuracy", f"{p['final_accuracy'] * 100:.1f}%"),
        ("communication", f"{p['communication_mib']:.2f} MiB"),
    ]
    for i, (label, value) in enumerate(metrics):
        col = i % 3
        row = i // 3
        x0 = 0.02 + col * 0.325
        y0 = 0.70 - row * 0.44
        ax_cards.add_patch(
            Rectangle(
                (x0, y0 - 0.22),
                0.29,
                0.30,
                transform=ax_cards.transAxes,
                facecolor=LIGHT,
                edgecolor=GRID,
                linewidth=0.65,
            )
        )
        ax_cards.text(
            x0 + 0.018,
            y0,
            label,
            transform=ax_cards.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color=MUTED,
        )
        ax_cards.text(
            x0 + 0.018,
            y0 - 0.115,
            value,
            transform=ax_cards.transAxes,
            ha="left",
            va="center",
            fontsize=7.1,
            color=INK,
            fontweight="bold",
        )
    panel_label(ax_cards, "d")
    save_fig(fig, "fig_acl_round2_01_partition_profile", profile_df)


def draw_learning_curves(
    comm_df: pd.DataFrame,
    server_eval_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.15], wspace=0.42)
    rounds = server_eval_df["round"].to_numpy()

    ax_acc = fig.add_subplot(gs[0, 0])
    chance = 0.20
    ax_acc.axhline(chance * 100, color=GRAY, lw=0.8, linestyle=(0, (2, 2)), label="5-way chance")
    ax_acc.plot(rounds, server_eval_df["accuracy"] * 100, color=BLUE, marker="o", lw=1.6, ms=4.4)
    for r, acc in zip(rounds, server_eval_df["accuracy"] * 100):
        ax_acc.text(r, acc + 0.9, f"{acc:.1f}", ha="center", va="bottom", fontsize=5.9, color=INK)
    ax_acc.set_xticks(rounds)
    ax_acc.set_xlabel("server round")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title("Server evaluation accuracy")
    ax_acc.set_ylim(18, max(32, server_eval_df["accuracy"].max() * 100 + 4))
    clean_axis(ax_acc)
    ax_acc.legend(loc="lower right", handlelength=1.4)
    panel_label(ax_acc, "a")

    ax_loss = fig.add_subplot(gs[0, 1])
    ax_loss.plot(rounds, server_eval_df["eval_loss"], color=ORANGE, marker="s", lw=1.6, ms=4.0)
    ax_loss.set_xticks(rounds)
    ax_loss.set_xlabel("server round")
    ax_loss.set_ylabel("eval loss")
    ax_loss.set_title("Server evaluation loss")
    ax_loss.set_ylim(0, server_eval_df["eval_loss"].max() * 1.22)
    for r, value in zip(rounds, server_eval_df["eval_loss"]):
        ax_loss.text(r, value + 0.010, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
    clean_axis(ax_loss)
    panel_label(ax_loss, "b")

    ax_train = fig.add_subplot(gs[0, 2])
    train_summary = (
        comm_df.groupby("round", sort=True)["train_loss"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    train_summary["se"] = train_summary["sd"].fillna(0) / np.sqrt(train_summary["n"])
    ax_train.plot(train_summary["round"], train_summary["mean"], color=TEAL, marker="D", lw=1.6, ms=4.0)
    ax_train.fill_between(
        train_summary["round"].to_numpy(dtype=float),
        (train_summary["mean"] - train_summary["se"]).to_numpy(dtype=float),
        (train_summary["mean"] + train_summary["se"]).to_numpy(dtype=float),
        color=TEAL,
        alpha=0.16,
        lw=0,
    )
    for client_id, group in comm_df.groupby("client_id"):
        ax_train.plot(
            group["round"],
            group["train_loss"],
            color=TEAL,
            alpha=0.22,
            lw=0.9,
            marker=".",
            ms=3.0,
        )
    ax_train.set_xticks(rounds)
    ax_train.set_xlabel("server round")
    ax_train.set_ylabel("train loss")
    ax_train.set_title("Client train loss")
    ax_train.set_ylim(comm_df["train_loss"].min() - 0.05, comm_df["train_loss"].max() + 0.06)
    clean_axis(ax_train)
    legend = [
        Line2D([0], [0], color=TEAL, marker="D", lw=1.6, label="mean"),
        Line2D([0], [0], color=TEAL, marker=".", lw=0.9, alpha=0.35, label="client"),
    ]
    ax_train.legend(handles=legend, loc="upper right", handlelength=1.6)
    panel_label(ax_train, "c")

    save_fig(fig, "fig_acl_round2_02_learning_curves", profile_df)


def draw_client_eval(
    client_eval_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.05], wspace=0.46)

    ax_heat = fig.add_subplot(gs[0, 0])
    mat = client_eval_df.pivot_table(index="client_id", columns="round", values="accuracy").sort_index()
    cmap = LinearSegmentedColormap.from_list("acc", ["#F6F3EF", "#E0B48A", ORANGE])
    im = ax_heat.imshow(mat.values * 100, aspect="auto", cmap=cmap, vmin=10, vmax=40)
    ax_heat.set_xticks(np.arange(mat.shape[1]))
    ax_heat.set_xticklabels([str(c) for c in mat.columns])
    ax_heat.set_yticks(np.arange(mat.shape[0]))
    ax_heat.set_yticklabels(mat.index)
    ax_heat.set_xlabel("server round")
    ax_heat.set_title("Client accuracy (%)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_heat.text(j, i, f"{mat.iat[i, j] * 100:.1f}", ha="center", va="center", fontsize=5.7)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    panel_label(ax_heat, "a")

    ax_final = fig.add_subplot(gs[0, 1])
    final_round = int(client_eval_df["round"].max())
    final = client_eval_df[client_eval_df["round"] == final_round].sort_values("accuracy")
    y = np.arange(len(final))
    ax_final.axvline(20, color=GRAY, lw=0.8, linestyle=(0, (2, 2)))
    ax_final.hlines(y, 20, final["accuracy"] * 100, color=BLUE, lw=1.1)
    ax_final.scatter(final["accuracy"] * 100, y, s=38, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3)
    for yi, (_, row) in enumerate(final.iterrows()):
        ax_final.text(row["accuracy"] * 100 + 0.8, yi, f"{row['accuracy'] * 100:.1f}", va="center", fontsize=6.0)
    ax_final.set_yticks(y)
    ax_final.set_yticklabels(final["client_id"])
    ax_final.set_xlabel("accuracy (%)")
    ax_final.set_title(f"Final client accuracy (round {final_round})")
    ax_final.set_xlim(10, max(42, final["accuracy"].max() * 100 + 7))
    clean_axis(ax_final, axis="x")
    panel_label(ax_final, "b")

    ax_latency = fig.add_subplot(gs[0, 2])
    latency = (
        client_eval_df.groupby("client_id", sort=True)["elapsed_sec_eval"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    x = np.arange(len(latency))
    ax_latency.bar(x, latency["mean"], yerr=latency["sd"], color=TEAL, width=0.56, capsize=2.5)
    ax_latency.set_xticks(x)
    ax_latency.set_xticklabels(latency["client_id"])
    ax_latency.set_ylabel("seconds")
    ax_latency.set_title("Eval latency per client")
    ax_latency.set_ylim(0, (latency["mean"] + latency["sd"].fillna(0)).max() * 1.20)
    clean_axis(ax_latency)
    panel_label(ax_latency, "c")

    save_fig(fig, "fig_acl_round2_03_client_eval", profile_df)


def draw_system_diagnostics(
    comm_df: pd.DataFrame,
    server_eval_df: pd.DataFrame,
    profile_df: pd.DataFrame,
) -> None:
    _ = server_eval_df
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.44)

    ax_drift = fig.add_subplot(gs[0, 0])
    for client_id, group in comm_df.groupby("client_id"):
        ax_drift.plot(group["round"], group["client_drift_l2"], marker="o", lw=1.1, ms=3.4, label=client_id)
    drift_mean = comm_df.groupby("round")["client_drift_l2"].mean()
    ax_drift.plot(drift_mean.index, drift_mean.values, color=INK, lw=1.7, marker="D", ms=4.0, label="mean")
    ax_drift.set_xticks(sorted(comm_df["round"].unique()))
    ax_drift.set_xlabel("server round")
    ax_drift.set_ylabel("L2 drift")
    ax_drift.set_title("Client update drift")
    ax_drift.set_ylim(comm_df["client_drift_l2"].min() - 0.015, comm_df["client_drift_l2"].max() + 0.015)
    clean_axis(ax_drift)
    ax_drift.legend(ncol=1, loc="lower right", handlelength=1.2, labelspacing=0.25)
    panel_label(ax_drift, "a")

    ax_comm = fig.add_subplot(gs[0, 1])
    by_round = comm_df.groupby("round", sort=True)[["uplink_mib", "downlink_mib"]].sum().reset_index()
    bottom = np.zeros(len(by_round))
    x = np.arange(len(by_round))
    ax_comm.bar(x, by_round["uplink_mib"], color=BLUE, width=0.56, label="uplink")
    bottom += by_round["uplink_mib"].to_numpy()
    ax_comm.bar(x, by_round["downlink_mib"], bottom=bottom, color=PALE_BLUE, width=0.56, edgecolor=BLUE, linewidth=0.6, label="downlink")
    ax_comm.set_xticks(x)
    ax_comm.set_xticklabels([str(r) for r in by_round["round"]])
    ax_comm.set_xlabel("server round")
    ax_comm.set_ylabel("MiB")
    ax_comm.set_title("Round communication")
    ax_comm.set_ylim(0, (by_round["uplink_mib"] + by_round["downlink_mib"]).max() * 1.20)
    clean_axis(ax_comm)
    ax_comm.legend(loc="upper right", handlelength=1.2)
    panel_label(ax_comm, "b")

    ax_time = fig.add_subplot(gs[0, 2])
    train_time = comm_df.pivot_table(index="client_id", columns="round", values="elapsed_sec_train").sort_index()
    im = ax_time.imshow(train_time.values, aspect="auto", cmap="Greys", vmin=train_time.values.min() * 0.96, vmax=train_time.values.max() * 1.04)
    ax_time.set_xticks(np.arange(train_time.shape[1]))
    ax_time.set_xticklabels([str(c) for c in train_time.columns])
    ax_time.set_yticks(np.arange(train_time.shape[0]))
    ax_time.set_yticklabels(train_time.index)
    ax_time.set_xlabel("server round")
    ax_time.set_title("Train elapsed time (s)")
    for i in range(train_time.shape[0]):
        for j in range(train_time.shape[1]):
            color = "white" if train_time.iat[i, j] > train_time.values.mean() else INK
            ax_time.text(j, i, f"{train_time.iat[i, j]:.1f}", ha="center", va="center", fontsize=5.7, color=color)
    cbar = fig.colorbar(im, ax=ax_time, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    panel_label(ax_time, "c")

    save_fig(fig, "fig_acl_round2_04_system_diagnostics", profile_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    configure_style()
    ensure_dirs()
    run_dir = args.run_dir if args.run_dir is not None else latest_eval_run(args.run_root)
    (
        comm_df,
        server_eval_df,
        client_eval_df,
        size_df,
        label_df,
        jurisdiction_df,
        profile_df,
    ) = load_run(run_dir)
    save_source_data(
        comm_df,
        server_eval_df,
        client_eval_df,
        size_df,
        label_df,
        jurisdiction_df,
        profile_df,
    )

    draw_partition_profile(size_df, label_df, jurisdiction_df, profile_df)
    draw_learning_curves(comm_df, server_eval_df, profile_df)
    draw_client_eval(client_eval_df, profile_df)
    draw_system_diagnostics(comm_df, server_eval_df, profile_df)

    print(f"Rendered ACL-style round-2 real figures from: {run_dir}")
    print(f"Figures: {FIG_DIR}")
    print(f"Source data: {DATA_DIR}")


if __name__ == "__main__":
    main()
