"""Rebuild manuscript Figure 2 from the verified 2026-09-08 panel results.

Panel A uses the reproducibly locked 16-panel patient-level curve. Panels B and
C retain the manuscript's primary-panel and feature-stability summaries.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

from _figure_style import BLACK, GREY, LIGHT, BLUE, ORANGE, RED, PALE_ORANGE


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/panel_knee_v5_verified_audit_20260908/final_package/results"
OUT = ROOT / "outputs/manuscript_panel_revision_20260908"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.6,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.3,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 600,
        "savefig.dpi": 600,
    }
)

# Log2(2) axis: the 16 evaluated panel sizes are kept as minor ticks, but only a
# sparsely labelled subset is printed so that the 800-2,000 region stays legible.
LABELLED_PANELS = (500, 1000, 2000, 5000, 10000)


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def label(ax: plt.Axes, text: str) -> None:
    ax.text(
        -0.18,
        1.08,
        text.upper(),
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def primary_row(ci: pd.DataFrame, panel: int, metric: str) -> pd.Series:
    return ci[(ci.panel_size == panel) & (ci.metric == metric)].iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ci = pd.read_csv(RESULTS / "patient_oof_primary_estimates_ci.tsv", sep="\t")
    paired = pd.read_csv(RESULTS / "paired_panel_comparison.tsv", sep="\t").iloc[0]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.85),
        gridspec_kw={"width_ratios": [1.62, 1.0, 1.0]},
    )

    # A: verified fine-grid curve.
    ax = axes[0]
    d = ci[ci.metric == "macro_f1"].sort_values("panel_size")
    x = d.panel_size.to_numpy(dtype=float)
    y = d.estimate_mean_across_repeats.to_numpy(dtype=float)
    lo = y - d.ci_low.to_numpy(dtype=float)
    hi = d.ci_high.to_numpy(dtype=float) - y
    ax.axvspan(1100, 11000, color=PALE_ORANGE, zorder=0)
    ax.errorbar(
        x,
        y,
        yerr=[lo, hi],
        fmt="o-",
        color=BLACK,
        lw=1.0,
        ms=2.7,
        ecolor=GREY,
        elinewidth=0.75,
        capsize=1.7,
        zorder=2,
    )
    for panel, colour in [(500, BLUE), (1000, ORANGE)]:
        row = d[d.panel_size == panel].iloc[0]
        ax.scatter(
            panel,
            row.estimate_mean_across_repeats,
            s=38,
            facecolor="white",
            edgecolor=colour,
            linewidth=1.35,
            zorder=4,
        )
    candidate = d[d.panel_size == 1500].iloc[0]
    ax.scatter(
        1500,
        candidate.estimate_mean_across_repeats,
        marker="D",
        s=34,
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.axvline(1500, color=ORANGE, lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.annotate(
        "1,500-CpG segmented candidate\n(not uniquely identified)",
        xy=(1500, candidate.estimate_mean_across_repeats),
        xytext=(2050, 0.9428),
        fontsize=5.3,
        color=RED,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.7},
        ha="left",
        va="top",
    )
    ax.text(
        0.985,
        0.055,
        "Broad saturation region",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=RED,
    )
    ax.set_xscale("log", base=2)
    # All 16 evaluated sizes keep a tick mark; only LABELLED_PANELS are annotated.
    # The log axis default locators are replaced explicitly, otherwise the minor
    # ticks inherit a formatter and print a second, overlapping row of labels.
    major = [v for v in LABELLED_PANELS if v in set(x)]
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{v:,}" for v in major]))
    ax.xaxis.set_minor_locator(FixedLocator(list(x)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2.4, width=0.6, color=GREY)
    ax.tick_params(axis="x", which="major", rotation=0)
    ax.set_ylim(0.90, 0.95)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Patient-level macro-F1 (95% CI)")
    ax.grid(axis="y", color=LIGHT, lw=0.65)
    clean(ax)
    label(ax, "a")
    ax.set_title("Patient-level panel-size curve", loc="left", fontweight="bold")

    # B: designated compact candidate and reference anchor.
    ax = axes[1]
    metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    for i, metric in enumerate(metrics):
        for panel, colour, offset in [(500, BLUE, -0.13), (1000, ORANGE, 0.13)]:
            row = primary_row(ci, panel, metric)
            value = row.estimate_mean_across_repeats
            ax.errorbar(
                i + offset,
                value,
                yerr=[[value - row.ci_low], [row.ci_high - value]],
                fmt="o",
                color=colour,
                capsize=2,
                ms=4,
                label=f"{panel:,} CpG" if i == 0 else None,
            )
    ax.set_xticks(range(3), ["Macro-F1", "Balanced\naccuracy", "Accuracy"])
    ax.set_ylim(0.90, 0.97)
    ax.set_ylabel("Patient-level performance (95% CI)")
    clean(ax)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")
    label(ax, "b")
    ax.set_title("Primary compact panels", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.02,
        rf"$\Delta$ macro-F1 = {paired.estimate:.4f}"
        + f"\n95% CI, {paired.ci_low:.4f} to {paired.ci_high:.4f}",
        transform=ax.transAxes,
        fontsize=5.2,
        va="bottom",
    )

    # C: previously reported fold-selection stability.
    ax = axes[2]
    metrics = ["Nogueira", "Jaccard", "RBO"]
    summary = {500: [0.673, 0.953, 0.971], 1000: [0.685, 0.970, 0.971]}
    xx = np.arange(3)
    bars_a = ax.bar(xx - 0.19, summary[500], 0.30, color=BLUE, label="500 CpG")
    bars_b = ax.bar(xx + 0.19, summary[1000], 0.30, color=ORANGE, label="1,000 CpG")
    # Value labels are staggered vertically: at a bar centre spacing of 0.38 axis
    # units the two labels of a pair would otherwise touch (e.g. "0.9710.971").
    for bars, dy in ((bars_a, 0.020), (bars_b, 0.052)):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + dy,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=5.0,
            )
    ax.set_xticks(xx, metrics)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Stability across 15 folds")
    clean(ax)
    ax.legend(frameon=False, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=2)
    label(ax, "c")
    ax.set_title("Feature reproducibility", loc="left", fontweight="bold")
    # Boxed so that the annotation stays readable, and anchored over the short
    # Nogueira bars: as free text at the lower right it sat on top of the
    # 0.971/0.971 RBO bars.
    ax.text(
        0.03,
        0.90,
        "Selected in every fold\n465/500  ·  958/1,000",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        color=GREY,
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.5, "pad": 1.8, "alpha": 0.95},
    )

    fig.subplots_adjust(wspace=0.51, bottom=0.27, left=0.075, right=0.985, top=0.89)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"figure2.{suffix}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(OUT / "figure2.pdf")


if __name__ == "__main__":
    main()
