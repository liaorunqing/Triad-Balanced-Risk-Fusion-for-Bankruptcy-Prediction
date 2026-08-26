"""Build Figure 1 for the triad-balanced bankruptcy-risk framework."""

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = (
    Path(__file__).resolve().parents[2]
    / "05_figures"
    / "manuscript"
    / "figure_01_revised"
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

COLORS = {
    "ink": "#23364A",
    "arrow": "#456786",
    "data": "#E8F0F5",
    "process": "#EEF3F8",
    "anchor": "#FFF1DF",
    "kernel": "#E7F3EC",
    "fusion": "#E2F2F1",
    "decision": "#FBEDEE",
    "search": "#E8EDFA",
    "objective": "#F3EAF7",
    "locked": "#F2F2F2",
    "lane": "#FAFBFC",
}


def box(ax, xy, width, height, text, fill, fontsize=7, linewidth=1.0):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=COLORS["ink"],
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=fontsize,
        linespacing=1.2,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, rad=0.0, linestyle="-", color=None, linewidth=1.25):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color or COLORS["arrow"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 183 mm x 108 mm, suitable for a two-column journal figure.
    fig, ax = plt.subplots(figsize=(183 / 25.4, 108 / 25.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Model-development lane.
    lane = FancyBboxPatch(
        (0.025, 0.35),
        0.95,
        0.57,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#AAB7C4",
        facecolor=COLORS["lane"],
        zorder=0,
    )
    ax.add_patch(lane)
    ax.text(
        0.045,
        0.875,
        "Model development uses fitting and validation data only",
        color=COLORS["ink"],
        fontsize=7.5,
        fontweight="bold",
        va="center",
    )

    box(ax, (0.045, 0.68), 0.12, 0.13, "Financial-ratio\ndata", COLORS["data"])
    box(
        ax,
        (0.195, 0.68),
        0.12,
        0.13,
        "Outer split\nand locked test set",
        COLORS["process"],
        fontsize=6.6,
    )
    box(
        ax,
        (0.345, 0.68),
        0.15,
        0.13,
        "Training-only\nimputation and scaling",
        COLORS["process"],
        fontsize=6.6,
    )
    box(
        ax,
        (0.535, 0.68),
        0.13,
        0.13,
        "Cross-fitted HistGB\nrisk anchor",
        COLORS["anchor"],
        fontsize=6.6,
    )
    box(
        ax,
        (0.53, 0.46),
        0.14,
        0.13,
        "Cost-sensitive\nlandmark RBF\ncorrection",
        COLORS["kernel"],
        fontsize=6.1,
    )
    box(
        ax,
        (0.72, 0.68),
        0.12,
        0.13,
        "Fused risk score",
        COLORS["fusion"],
        fontsize=6.8,
    )
    box(
        ax,
        (0.79, 0.44),
        0.16,
        0.15,
        "Validation triad\nobjective and\nthreshold selection",
        COLORS["decision"],
        fontsize=6.5,
    )

    arrow(ax, (0.165, 0.745), (0.195, 0.745))
    arrow(ax, (0.315, 0.745), (0.345, 0.745))
    arrow(ax, (0.495, 0.745), (0.535, 0.745))
    arrow(ax, (0.665, 0.745), (0.72, 0.745))
    arrow(ax, (0.495, 0.71), (0.53, 0.54), rad=0.06)
    arrow(ax, (0.60, 0.68), (0.60, 0.59), linestyle="--", linewidth=1.0)
    ax.text(
        0.607,
        0.635,
        "risk feature",
        fontsize=5.8,
        color=COLORS["arrow"],
        va="center",
    )
    arrow(ax, (0.67, 0.525), (0.72, 0.70), rad=-0.08)
    arrow(ax, (0.78, 0.68), (0.84, 0.59), rad=-0.05)

    # Optimizer loop. The compact labels prevent text from competing with arrows.
    box(
        ax,
        (0.275, 0.46),
        0.17,
        0.13,
        "PBMSBAINGO search\nhyperparameters\nand feature mask",
        COLORS["search"],
        fontsize=6.3,
    )
    arrow(ax, (0.445, 0.525), (0.53, 0.525))
    arrow(ax, (0.79, 0.47), (0.445, 0.485), rad=-0.11, linestyle="--", linewidth=1.0)
    ax.text(
        0.625,
        0.415,
        "validation feedback",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["arrow"],
    )

    # Locked held-out evaluation lane.
    ax.plot([0.025, 0.975], [0.30, 0.30], color="#AAB7C4", lw=0.8)
    ax.text(
        0.045,
        0.255,
        "Held-out evaluation",
        color=COLORS["ink"],
        fontsize=7.5,
        fontweight="bold",
        va="center",
    )
    box(
        ax,
        (0.25, 0.105),
        0.18,
        0.10,
        "Locked test features\nlabels hidden during\nmodel selection",
        COLORS["locked"],
        fontsize=6.5,
    )
    box(
        ax,
        (0.53, 0.105),
        0.18,
        0.10,
        "Apply finalized pipeline\nwith the fixed\nthreshold",
        COLORS["fusion"],
        fontsize=6.5,
    )
    box(
        ax,
        (0.79, 0.105),
        0.16,
        0.10,
        "Final predictions and\nreported test metrics",
        COLORS["decision"],
        fontsize=6.5,
    )
    arrow(ax, (0.255, 0.68), (0.33, 0.205), rad=0.05, linestyle="--", linewidth=1.0)
    arrow(ax, (0.43, 0.155), (0.53, 0.155))
    arrow(ax, (0.71, 0.155), (0.79, 0.155))
    arrow(ax, (0.87, 0.44), (0.62, 0.205), rad=-0.10, linestyle="--", linewidth=1.0)
    ax.text(
        0.50,
        0.045,
        "The same source-only fitting logic is retained in leave-one-region-out transfer tasks.",
        ha="center",
        va="center",
        fontsize=6.4,
        color="#526577",
    )

    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUT.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
