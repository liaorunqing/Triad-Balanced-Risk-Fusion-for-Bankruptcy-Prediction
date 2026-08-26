from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "04_results" / "03_manuscript_tables"
FIGURES = ROOT / "05_figures" / "manuscript"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Triad_risk_fusion": "#0B7285",
    "HistGB_fair": "#4C6EF5",
    "Random_forest": "#F59F00",
    "EasyEnsemble_reference": "#AE3EC9",
    "Balanced_forest_reference": "#2F9E44",
    "Balanced_logistic": "#868E96",
    "Extra_trees": "#E8590C",
}
LABELS = {
    "Triad_risk_fusion": "Triad risk fusion",
    "HistGB_fair": "HistGB (fair)",
    "Random_forest": "Random forest",
    "EasyEnsemble_reference": "EasyEnsemble-style",
    "Balanced_forest_reference": "Balanced forest ref.",
    "Balanced_logistic": "Balanced logistic",
    "Extra_trees": "Extra Trees",
}
METRICS = ["mcc", "sensitivity", "specificity", "precision", "f2", "ap"]
METRIC_LABELS = ["MCC", "Sensitivity", "Specificity", "Precision", r"$F_2$", "AP"]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def internal_tradeoff(blocks: pd.DataFrame) -> None:
    selected = blocks.loc[
        blocks["method"].isin(["Triad_risk_fusion", "HistGB_fair"])
    ]
    means = selected.groupby("method")[METRICS].mean()
    stds = selected.groupby("method")[METRICS].std(ddof=1)
    x = np.arange(len(METRICS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for offset, method in [(-width / 2, "Triad_risk_fusion"), (width / 2, "HistGB_fair")]:
        ax.bar(
            x + offset,
            means.loc[method].to_numpy(),
            width,
            yerr=stds.loc[method].to_numpy(),
            capsize=2.5,
            color=COLORS[method],
            label=LABELS[method],
            alpha=0.92,
        )
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean test value across 50 blocks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    save(fig, "figure_02_revised.png")


def dataset_delta() -> None:
    pairs = pd.read_csv(RESULTS / "triad_vs_fair_histgb_dataset_pairs.csv")
    matrix = pairs.pivot(index="dataset", columns="metric", values="difference")
    matrix = matrix.reindex(columns=METRICS)
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    bound = np.nanmax(np.abs(matrix.to_numpy()))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)), METRIC_LABELS)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Triad minus fair HistGB at the dataset level")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean difference")
    fig.tight_layout()
    save(fig, "figure_03_revised.png")


def block_distributions(blocks: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12.8, 3.4), sharey=False)
    for axis, metric, label in zip(axes, METRICS, METRIC_LABELS):
        values = [
            blocks.loc[blocks["method"] == method, metric].to_numpy()
            for method in ["Triad_risk_fusion", "HistGB_fair"]
        ]
        box = axis.boxplot(values, patch_artist=True, widths=0.58, showfliers=False)
        for patch, method in zip(box["boxes"], ["Triad_risk_fusion", "HistGB_fair"]):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.72)
        axis.set_xticks([1, 2], ["Triad", "HistGB"], rotation=35, ha="right")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Paired test-block distributions under the common protocol", y=1.02)
    fig.tight_layout()
    save(fig, "figure_04_revised.png")


def cost_profiles() -> None:
    costs = pd.read_csv(RESULTS / "fixed_operating_cost_summary.csv")
    methods = [
        "Triad_risk_fusion",
        "HistGB_fair",
        "Random_forest",
        "EasyEnsemble_reference",
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for method in methods:
        group = costs.loc[costs["method"] == method].sort_values("cost_ratio")
        ax.plot(
            group["cost_ratio"],
            group["mean"],
            marker="o",
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel(r"Relative false-negative cost $C_{FN}/C_{FP}$")
    ax.set_ylabel("Mean normalized cost at the fixed operating point")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "figure_05_revised.png")


def fpr_fnr(blocks: pd.DataFrame) -> None:
    methods = list(COLORS)
    means = blocks.groupby("method")[["sensitivity", "specificity"]].mean()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for method in methods:
        x = 1.0 - means.loc[method, "specificity"]
        y = 1.0 - means.loc[method, "sensitivity"]
        ax.scatter(x, y, s=58, color=COLORS[method], edgecolor="white", linewidth=0.7)
        ax.annotate(LABELS[method], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Mean false-positive rate")
    ax.set_ylabel("Mean false-negative rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "figure_06_revised.png")


def cross_region_probability() -> None:
    summary = pd.read_csv(RESULTS / "cross_region_extended_summary.csv")
    methods = [
        "Balanced_logistic",
        "HistGB",
        "KELM_score_fusion",
        "Full_PBMSBAINGO",
        "Full_random_search",
    ]
    labels = ["Balanced logistic", "HistGB", "Score fusion", "Full model", "Random search"]
    metrics = ["brier", "log_loss", "ece_10", "absolute_intercept"]
    titles = ["Brier score", "Log loss", "ECE (10 bins)", "Absolute calibration intercept"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.3))
    for axis, metric, title in zip(axes.ravel(), metrics, titles):
        metric_rows = summary.loc[summary["metric"] == metric].set_index("method")
        values = [metric_rows.loc[method, "mean"] for method in methods]
        errors = np.asarray([metric_rows.loc[method, "sd"] for method in methods])
        values_array = np.asarray(values)
        asymmetric_errors = np.vstack([np.minimum(errors, values_array), errors])
        axis.bar(
            np.arange(len(methods)),
            values,
            yerr=asymmetric_errors,
            capsize=2.5,
            color=["#868E96", "#4C6EF5", "#12B886", "#0B7285", "#F59F00"],
        )
        axis.set_xticks(np.arange(len(methods)), labels, rotation=28, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Cross-regional probability-quality diagnostics across nine tasks", y=1.01)
    fig.tight_layout()
    save(fig, "figure_07_revised.png")


def main() -> None:
    blocks = pd.read_csv(RESULTS / "internal_revised_blocks.csv")
    internal_tradeoff(blocks)
    dataset_delta()
    block_distributions(blocks)
    cost_profiles()
    fpr_fnr(blocks)
    cross_region_probability()
    print(f"Saved revised figures to {FIGURES}")


if __name__ == "__main__":
    main()
