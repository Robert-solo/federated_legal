"""Generate ACL-style figures from the latest real federated pilot run.

The script reads the measured Flower/PEFT pilot artifacts and renders only
diagnostics supported by those artifacts. It does not invent accuracy,
citation, hallucination, or multi-round metrics.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "real_flower_peft"
FIG_DIR = PAPER_DIR / "figures" / "experiments_acl_real"
DATA_DIR = PAPER_DIR / "tables" / "experiment_source_data_acl_real"

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
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
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


def latest_run(run_root: Path) -> Path:
    candidates = sorted(
        [p for p in run_root.glob("*") if (p / "real_experiment_report.json").exists()],
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No real_experiment_report.json found under {run_root}")
    return candidates[-1]


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report = read_json(run_dir / "real_experiment_report.json")
    manifest = read_json(run_dir / "manifest.json")
    comm_rows = read_jsonl(run_dir / "communication_log.jsonl")
    round_rows = read_jsonl(run_dir / "server_rounds.jsonl")

    clients: list[dict[str, Any]] = []
    for row in comm_rows:
        metrics = row.get("metrics", {})
        clients.append(
            {
                "client_id": row.get("client_id", "client"),
                "train_loss": float(metrics.get("train_loss", np.nan)),
                "elapsed_sec": float(metrics.get("elapsed_sec", np.nan)),
                "num_examples": int(float(metrics.get("num_examples", np.nan))),
                "payload_bytes": int(row.get("num_bytes", metrics.get("num_bytes", 0))),
                "payload_mib": float(row.get("num_bytes", metrics.get("num_bytes", 0)))
                / 1024**2,
                "timestamp": float(row.get("timestamp", np.nan)),
            }
        )
    client_df = pd.DataFrame(clients).sort_values("client_id").reset_index(drop=True)
    if client_df.empty:
        raise ValueError(f"No communication rows found in {run_dir}")
    client_df["throughput_mib_sec"] = client_df["payload_mib"] / client_df["elapsed_sec"]
    client_df["loss_centered"] = client_df["train_loss"] - client_df["train_loss"].mean()

    shards = []
    for path in sorted((run_dir / "client_shards").glob("client_*.jsonl")):
        shards.append({"client_id": path.stem, "shard_records": count_jsonl(path)})
    shard_df = pd.DataFrame(shards)
    if not shard_df.empty:
        client_df = client_df.merge(shard_df, on="client_id", how="left")
    else:
        client_df["shard_records"] = client_df["num_examples"]

    rounds = []
    for row in round_rows:
        rounds.append(
            {
                "round": int(row.get("round", 0)),
                "num_results": int(row.get("num_results", 0)),
                "num_failures": int(row.get("num_failures", 0)),
                "timestamp": float(row.get("timestamp", np.nan)),
            }
        )
    round_df = pd.DataFrame(rounds)

    train_count = count_jsonl(run_dir / "train.jsonl")
    eval_count = count_jsonl(run_dir / "eval.jsonl")
    profile = {
        "run_name": report.get("run_name", run_dir.name),
        "dataset": report.get(
            "dataset",
            f"{manifest.get('dataset', '')}/{manifest.get('dataset_config', '')}",
        ),
        "model_name": report.get("model_name", manifest.get("model_name", "")),
        "num_clients": int(report.get("num_clients", manifest.get("num_clients", 0))),
        "num_rounds": int(report.get("num_rounds", report.get("server_rounds", 0))),
        "server_rounds": int(report.get("server_rounds", len(round_df))),
        "train_records_manifest": int(manifest.get("train_records", 0)),
        "eval_records_manifest": int(manifest.get("eval_records", 0)),
        "train_records_observed": train_count,
        "eval_records_observed": eval_count,
        "mean_train_loss_report": float(report.get("mean_train_loss", np.nan)),
        "mean_train_loss_clients": float(client_df["train_loss"].mean()),
        "sd_train_loss_clients": float(client_df["train_loss"].std(ddof=1)),
        "communication_bytes_report": int(report.get("communication_bytes", 0)),
        "communication_bytes_clients": int(client_df["payload_bytes"].sum()),
        "communication_mib_clients": float(client_df["payload_mib"].sum()),
        "communication_events": int(report.get("communication_events", len(client_df))),
        "parallel_wall_clock_sec": float(client_df["elapsed_sec"].max()),
        "sum_client_time_sec": float(client_df["elapsed_sec"].sum()),
        "bytes_per_example": float(client_df["payload_bytes"].sum())
        / max(float(client_df["num_examples"].sum()), 1.0),
        "run_dir": str(run_dir),
    }
    profile_df = pd.DataFrame([profile])
    return client_df, round_df, profile_df


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_source_data(
    client_df: pd.DataFrame, round_df: pd.DataFrame, profile_df: pd.DataFrame
) -> None:
    client_df.to_csv(DATA_DIR / "acl_real_client_diagnostics.csv", index=False)
    round_df.to_csv(DATA_DIR / "acl_real_server_rounds.csv", index=False)
    profile_df.to_csv(DATA_DIR / "acl_real_run_profile.csv", index=False)


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
    profile = profile_df.iloc[0]
    note = (
        f"REAL PILOT RUN: {profile['dataset']} | "
        f"{int(profile['num_clients'])} clients | "
        f"{int(profile['server_rounds'])} server round | "
        "accuracy/citation metrics not emitted by this run"
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


def draw_run_profile(
    client_df: pd.DataFrame, round_df: pd.DataFrame, profile_df: pd.DataFrame
) -> None:
    profile = profile_df.iloc[0]
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.85))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.15, 1.0], hspace=0.62, wspace=0.42)

    ax_data = fig.add_subplot(gs[0, 0])
    counts = pd.DataFrame(
        {
            "split": ["Train", "Eval"],
            "records": [
                profile["train_records_observed"],
                profile["eval_records_observed"],
            ],
        }
    )
    bars = ax_data.barh(counts["split"], counts["records"], color=[BLUE, GRAY], height=0.45)
    for bar in bars:
        ax_data.text(
            bar.get_width() + max(counts["records"]) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=INK,
        )
    ax_data.set_xlabel("records")
    ax_data.set_title("Dataset footprint")
    ax_data.set_xlim(0, max(counts["records"]) * 1.18)
    clean_axis(ax_data, axis="x")
    panel_label(ax_data, "a")

    ax_shards = fig.add_subplot(gs[0, 1])
    x = np.arange(len(client_df))
    ax_shards.bar(
        x,
        client_df["shard_records"],
        color=TEAL,
        width=0.58,
        edgecolor="white",
        linewidth=0.6,
    )
    for xi, value in zip(x, client_df["shard_records"]):
        ax_shards.text(xi, value + 3, f"{int(value)}", ha="center", va="bottom", fontsize=6.3)
    ax_shards.set_xticks(x)
    ax_shards.set_xticklabels(client_df["client_id"], rotation=0)
    ax_shards.set_ylabel("records")
    ax_shards.set_title("Client shards")
    ax_shards.set_ylim(0, max(client_df["shard_records"]) * 1.22)
    clean_axis(ax_shards)
    panel_label(ax_shards, "b")

    ax_round = fig.add_subplot(gs[0, 2])
    successes = int(round_df["num_results"].sum()) if not round_df.empty else len(client_df)
    failures = int(round_df["num_failures"].sum()) if not round_df.empty else 0
    ax_round.barh([0], [successes], color=TEAL, height=0.38, label="completed")
    if failures:
        ax_round.barh([0], [failures], left=[successes], color=RED, height=0.38, label="failed")
    ax_round.set_xlim(0, max(successes + failures, 1) * 1.18)
    ax_round.set_yticks([0])
    ax_round.set_yticklabels(["round 1"])
    ax_round.set_xlabel("client updates")
    ax_round.set_title("Server participation")
    ax_round.text(
        successes + 0.07,
        0,
        f"{successes} ok, {failures} fail",
        va="center",
        ha="left",
        fontsize=6.3,
        color=INK,
    )
    clean_axis(ax_round, axis="x")
    panel_label(ax_round, "c")

    ax_metrics = fig.add_subplot(gs[1, :])
    ax_metrics.axis("off")
    metrics = [
        ("model", str(profile["model_name"]).replace("Qwen/", "")),
        ("dataset", str(profile["dataset"])),
        ("mean train loss", f"{profile['mean_train_loss_clients']:.3f}"),
        ("payload", f"{profile['communication_mib_clients']:.2f} MiB"),
        ("wall-clock proxy", f"{profile['parallel_wall_clock_sec']:.2f} s"),
        ("bytes/example", f"{profile['bytes_per_example'] / 1024:.1f} KiB"),
    ]
    for i, (label, value) in enumerate(metrics):
        col = i % 3
        row = i // 3
        x0 = 0.02 + col * 0.325
        y0 = 0.70 - row * 0.44
        ax_metrics.add_patch(
            Rectangle(
                (x0, y0 - 0.22),
                0.29,
                0.30,
                transform=ax_metrics.transAxes,
                facecolor=LIGHT,
                edgecolor=GRID,
                linewidth=0.65,
            )
        )
        ax_metrics.text(
            x0 + 0.018,
            y0,
            label,
            transform=ax_metrics.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color=MUTED,
        )
        ax_metrics.text(
            x0 + 0.018,
            y0 - 0.115,
            value,
            transform=ax_metrics.transAxes,
            ha="left",
            va="center",
            fontsize=7.1,
            color=INK,
            fontweight="bold",
        )
    panel_label(ax_metrics, "d")

    save_fig(fig, "fig_acl_real01_run_profile", profile_df)


def draw_client_training(
    client_df: pd.DataFrame, round_df: pd.DataFrame, profile_df: pd.DataFrame
) -> None:
    _ = round_df
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.05, 1.05], wspace=0.46)
    x = np.arange(len(client_df))
    labels = client_df["client_id"].to_list()

    ax_loss = fig.add_subplot(gs[0, 0])
    mean_loss = client_df["train_loss"].mean()
    sd_loss = client_df["train_loss"].std(ddof=1)
    ax_loss.axhspan(mean_loss - sd_loss, mean_loss + sd_loss, color=BLUE, alpha=0.10, lw=0)
    ax_loss.axhline(mean_loss, color=INK, lw=0.9, linestyle=(0, (3, 2)))
    ax_loss.vlines(x, mean_loss, client_df["train_loss"], color=BLUE, lw=1.0, alpha=0.85)
    ax_loss.scatter(x, client_df["train_loss"], s=38, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3)
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(labels)
    ax_loss.set_ylabel("train loss")
    ax_loss.set_title("Local objective after one round")
    ymin = client_df["train_loss"].min() - 0.045
    ymax = client_df["train_loss"].max() + 0.045
    ax_loss.set_ylim(ymin, ymax)
    ax_loss.text(
        0.03,
        0.94,
        f"mean={mean_loss:.3f}; sd={sd_loss:.3f}",
        transform=ax_loss.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    clean_axis(ax_loss)
    panel_label(ax_loss, "a")

    ax_time = fig.add_subplot(gs[0, 1])
    mean_time = client_df["elapsed_sec"].mean()
    ax_time.bar(
        x,
        client_df["elapsed_sec"],
        color=TEAL,
        width=0.56,
        edgecolor="white",
        linewidth=0.6,
    )
    ax_time.axhline(mean_time, color=INK, lw=0.85, linestyle=(0, (3, 2)))
    for xi, value in zip(x, client_df["elapsed_sec"]):
        ax_time.text(xi, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=6.0)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.set_ylabel("seconds")
    ax_time.set_title("Client training time")
    ax_time.set_ylim(0, client_df["elapsed_sec"].max() * 1.18)
    clean_axis(ax_time)
    panel_label(ax_time, "b")

    ax_scatter = fig.add_subplot(gs[0, 2])
    sizes = np.clip(client_df["num_examples"] / client_df["num_examples"].max() * 115, 45, 115)
    ax_scatter.scatter(
        client_df["elapsed_sec"],
        client_df["train_loss"],
        s=sizes,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.95,
    )
    for _, row in client_df.iterrows():
        ax_scatter.text(
            row["elapsed_sec"] + 0.004,
            row["train_loss"] + 0.004,
            row["client_id"].replace("client_", "c"),
            fontsize=5.9,
            color=INK,
        )
    ax_scatter.set_xlabel("elapsed seconds")
    ax_scatter.set_ylabel("train loss")
    ax_scatter.set_title("Loss--time profile")
    ax_scatter.set_xlim(client_df["elapsed_sec"].min() - 0.035, client_df["elapsed_sec"].max() + 0.065)
    ax_scatter.set_ylim(ymin, ymax)
    clean_axis(ax_scatter)
    panel_label(ax_scatter, "c")

    save_fig(fig, "fig_acl_real02_client_training", profile_df)


def draw_communication(
    client_df: pd.DataFrame, round_df: pd.DataFrame, profile_df: pd.DataFrame
) -> None:
    _ = round_df
    profile = profile_df.iloc[0]
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.05], wspace=0.47)
    x = np.arange(len(client_df))
    labels = client_df["client_id"].to_list()

    ax_payload = fig.add_subplot(gs[0, 0])
    ax_payload.bar(
        x,
        client_df["payload_mib"],
        color=BLUE,
        width=0.56,
        edgecolor="white",
        linewidth=0.6,
    )
    for xi, value in zip(x, client_df["payload_mib"]):
        ax_payload.text(xi, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=6.0)
    ax_payload.set_xticks(x)
    ax_payload.set_xticklabels(labels)
    ax_payload.set_ylabel("MiB")
    ax_payload.set_title("Adapter payload per client")
    ax_payload.set_ylim(0, client_df["payload_mib"].max() * 1.20)
    clean_axis(ax_payload)
    panel_label(ax_payload, "a")

    ax_cum = fig.add_subplot(gs[0, 1])
    order = client_df.sort_values("timestamp").reset_index(drop=True)
    cum = order["payload_mib"].cumsum()
    ax_cum.step(np.arange(1, len(cum) + 1), cum, where="mid", color=TEAL, lw=1.6)
    ax_cum.scatter(np.arange(1, len(cum) + 1), cum, color=TEAL, s=24, zorder=3)
    ax_cum.set_xticks(np.arange(1, len(cum) + 1))
    ax_cum.set_xlabel("received update")
    ax_cum.set_ylabel("MiB")
    ax_cum.set_title("Cumulative traffic")
    ax_cum.set_ylim(0, cum.max() * 1.16)
    ax_cum.text(
        0.05,
        0.93,
        f"total={profile['communication_mib_clients']:.2f} MiB",
        transform=ax_cum.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    clean_axis(ax_cum)
    panel_label(ax_cum, "b")

    ax_eff = fig.add_subplot(gs[0, 2])
    metrics = pd.DataFrame(
        {
            "metric": ["payload/client", "payload/example", "throughput"],
            "value": [
                client_df["payload_mib"].mean(),
                profile["bytes_per_example"] / 1024,
                client_df["throughput_mib_sec"].mean(),
            ],
            "unit": ["MiB", "KiB", "MiB/s"],
            "color": [BLUE, ORANGE, TEAL],
        }
    )
    y = np.arange(len(metrics))
    ax_eff.barh(y, metrics["value"], color=metrics["color"], height=0.45)
    for yi, value, unit in zip(y, metrics["value"], metrics["unit"]):
        ax_eff.text(value * 1.02, yi, f"{value:.2f} {unit}", va="center", ha="left", fontsize=6.1)
    ax_eff.set_yticks(y)
    ax_eff.set_yticklabels(metrics["metric"])
    ax_eff.invert_yaxis()
    ax_eff.set_xlabel("measured value")
    ax_eff.set_title("Communication efficiency")
    ax_eff.set_xlim(0, metrics["value"].max() * 1.32)
    clean_axis(ax_eff, axis="x")
    panel_label(ax_eff, "c")

    save_fig(fig, "fig_acl_real03_communication", profile_df)


def draw_round_status(
    client_df: pd.DataFrame, round_df: pd.DataFrame, profile_df: pd.DataFrame
) -> None:
    profile = profile_df.iloc[0]
    fig = plt.figure(figsize=(WIDTH_SINGLE, 2.35))
    ax = fig.add_subplot(111)
    success = int(round_df["num_results"].sum()) if not round_df.empty else len(client_df)
    failure = int(round_df["num_failures"].sum()) if not round_df.empty else 0
    total = max(success + failure, 1)
    y = [0]
    ax.barh(y, [success], color=TEAL, height=0.34, label="successful updates")
    if failure:
        ax.barh(y, [failure], left=[success], color=RED, height=0.34, label="failed updates")
    ax.set_yticks(y)
    ax.set_yticklabels(["server round 1"])
    ax.set_xlim(0, total * 1.25)
    ax.set_xlabel("client updates")
    ax.set_title("Round completion in the real pilot")
    ax.text(success + 0.08, 0, f"{success}/{total} completed", va="center", ha="left", fontsize=6.4)
    ax.text(
        0.00,
        -0.64,
        (
            f"{int(profile['num_clients'])} clients, "
            f"{int(profile['communication_events'])} communication events, "
            f"{profile['communication_mib_clients']:.2f} MiB total payload"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=MUTED,
    )
    clean_axis(ax, axis="x")
    save_fig(fig, "fig_acl_real04_round_status", profile_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Directory containing real experiment run folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific real experiment run directory. Defaults to latest under --run-root.",
    )
    args = parser.parse_args()

    configure_style()
    ensure_dirs()
    run_dir = args.run_dir if args.run_dir is not None else latest_run(args.run_root)
    client_df, round_df, profile_df = load_run(run_dir)
    save_source_data(client_df, round_df, profile_df)

    draw_run_profile(client_df, round_df, profile_df)
    draw_client_training(client_df, round_df, profile_df)
    draw_communication(client_df, round_df, profile_df)
    draw_round_status(client_df, round_df, profile_df)

    print(f"Rendered ACL-style real pilot figures from: {run_dir}")
    print(f"Figures: {FIG_DIR}")
    print(f"Source data: {DATA_DIR}")


if __name__ == "__main__":
    main()
