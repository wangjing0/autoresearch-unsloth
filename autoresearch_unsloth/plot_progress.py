"""
Plot autoresearch-unsloth experiment progress from results_unsloth.tsv.
Mirrors the style of the root autoresearch progress.png.

Usage:
    uv run autoresearch_unsloth/plot_progress.py
    uv run autoresearch_unsloth/plot_progress.py --results path/to/results_unsloth.tsv
    uv run autoresearch_unsloth/plot_progress.py --out autoresearch_unsloth/progress_unsloth.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd

DEFAULT_RESULTS = os.path.join(os.path.dirname(__file__), "results_unsloth.tsv")
DEFAULT_OUT     = os.path.join(os.path.dirname(__file__), "progress_unsloth.png")


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df = df[df["status"].isin(["keep", "discard", "crash"])].reset_index(drop=True)
    df["experiment"] = range(len(df))
    return df


def running_best(df: pd.DataFrame) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    best = float("inf")
    for _, row in df.iterrows():
        if row["status"] == "keep" and row["eval_loss"] < best:
            best = row["eval_loss"]
        xs.append(row["experiment"])
        ys.append(best if best < float("inf") else None)
    return xs, ys


def plot(df: pd.DataFrame, out: str):
    kept    = df[df["status"] == "keep"]
    non_kept = df[df["status"] != "keep"]

    n_total = len(df)
    n_kept  = len(kept)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Discarded / crash dots
    ax.scatter(
        non_kept["experiment"], non_kept["eval_loss"],
        color="#cccccc", s=18, zorder=2, label="Discarded",
    )

    # Running best step line
    rb_x, rb_y = running_best(df)
    valid = [(x, y) for x, y in zip(rb_x, rb_y) if y is not None]
    if valid:
        vx, vy = zip(*valid)
        ax.step(vx, vy, where="post", color="#2ecc71", linewidth=1.5, zorder=3)

    # Kept dots
    ax.scatter(
        kept["experiment"], kept["eval_loss"],
        color="#2ecc71", s=40, zorder=4, label="Kept",
    )

    # Labels on kept experiments
    for _, row in kept.iterrows():
        label = str(row["description"])[:40]
        ax.annotate(
            label,
            xy=(row["experiment"], row["eval_loss"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#1a8a4a",
            rotation=30,
            ha="left",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Eval Loss (lower is better)", fontsize=11)
    ax.set_title(
        f"Autoresearch-Unsloth Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        fontsize=12,
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc",
                       markersize=7, label="Discarded"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
                       markersize=7, label="Kept"),
            plt.Line2D([0], [0], color="#2ecc71", linewidth=1.5, label="Running best"),
        ],
        loc="upper right", fontsize=9,
    )

    ax.invert_yaxis()   # lower eval_loss is better, so best results appear higher on chart
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}  ({n_total} experiments, {n_kept} kept)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Path to results_unsloth.tsv")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output image path")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"No results file found at {args.results}")
        print("Run some experiments first, then re-run this script.")
        raise SystemExit(1)

    df = load_results(args.results)
    if df.empty:
        print("Results file contains no completed experiments yet.")
        raise SystemExit(1)

    plot(df, args.out)
