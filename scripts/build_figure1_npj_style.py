#!/usr/bin/env python3
"""Build a workflow-only Figure 1 in the visual grammar of the npj references."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figure_style import (  # noqa: E402
    BLACK,
    GREY,
    BLUE,
    ORANGE,
    GREEN,
    BLUE_TINT as BLUE_LIGHT,
    ORANGE_TINT as ORANGE_LIGHT,
    GREEN_TINT as GREEN_LIGHT,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "manuscript_figures_npj_fig1"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Journal raster floor: the PDF backend otherwise embeds rasters at 100 dpi.
    "figure.dpi": 600,
    "savefig.dpi": 600,
})


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.0, color=GREY))


def box(ax, x, y, w, h, title, body, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=.012,rounding_size=.014",
                                facecolor=face, edgecolor=edge, linewidth=1.0))
    ax.text(x + w / 2, y + h - .055, title, ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=edge)
    ax.text(x + w / 2, y + h / 2 - .035, body, ha="center", va="center",
            fontsize=5.7, color=BLACK, linespacing=1.25)


def main():
    fig, axes = plt.subplots(3, 1, figsize=(7.15, 4.2))
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

    ax = axes[0]
    ax.text(.005, .96, "A", fontsize=11, fontweight="bold", va="top")
    ax.text(.04, .94, "Development cohort and patient-level evaluation",
            fontsize=8, fontweight="bold", va="top")
    box(ax, .04, .16, .22, .60, "TCGA 450K matrix",
        "9,065 tumours\n8,923 patients\n33 cancer classes", BLUE, BLUE_LIGHT)
    arrow(ax, .27, .46, .35, .46)
    box(ax, .36, .16, .25, .60, "Repeated cross-validation",
        "5 folds × 3 repeats\npatients kept within folds\n8,923 OOF predictions/repeat",
        BLUE, BLUE_LIGHT)
    arrow(ax, .62, .46, .70, .46)
    box(ax, .71, .16, .25, .60, "Fold-specific workflow",
        "training-fold imputation\nand scaling\nvariance-ranked 500/1,000 CpGs",
        BLUE, BLUE_LIGHT)

    ax = axes[1]
    ax.text(.005, .96, "B", fontsize=11, fontweight="bold", va="top")
    ax.text(.04, .94, "Panel locking and final model fitting",
            fontsize=8, fontweight="bold", va="top")
    box(ax, .04, .16, .23, .60, "15 fold selections",
        "ranked by recurrence\nand mean within-fold rank", ORANGE, ORANGE_LIGHT)
    arrow(ax, .28, .46, .36, .46)
    box(ax, .37, .16, .23, .60, "Locked consensus panels",
        "500-CpG compact panel\n1,000-CpG reference panel", ORANGE, ORANGE_LIGHT)
    ax.plot([.64, .64], [.12, .82], color=GREY, lw=.9, ls=(0, (3, 2)))
    ax.text(.64, .08, "parameters locked", ha="center", va="top",
            fontsize=5.4, color=GREY)
    arrow(ax, .61, .46, .70, .46)
    box(ax, .71, .16, .25, .60, "Full-development models",
        "one multinomial model/panel\nfitted to all 9,065 tumours\nexternal data used for evaluation",
        ORANGE, ORANGE_LIGHT)

    ax = axes[2]
    ax.text(.005, .96, "C", fontsize=11, fontweight="bold", va="top")
    ax.text(.04, .94, "Evaluation scope", fontsize=8, fontweight="bold", va="top")
    box(ax, .04, .16, .27, .60, "Independent 450K cohorts",
        "4 GEO cohorts · 307 samples\nunrestricted, six-state\nand positive-class endpoints",
        GREEN, GREEN_LIGHT)
    box(ax, .365, .16, .27, .60, "Processed EPIC cohorts",
        "6 GEO cohorts · 460 patients\npositive-class recall\ncross-platform transportability",
        GREEN, GREEN_LIGHT)
    box(ax, .69, .16, .27, .60, "Robustness analyses",
        "normal-profile admixture\nout-of-fold probe loss\n747 retained normal tissues",
        GREEN, GREEN_LIGHT)

    fig.subplots_adjust(left=.02, right=.99, top=.99, bottom=.03, hspace=.08)
    for suffix, kwargs in (("pdf", {"dpi": 600}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"figure1_study_design.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Wrote {OUT / 'figure1_study_design.pdf'}")


if __name__ == "__main__":
    main()
