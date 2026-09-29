from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _figure_style import (  # noqa: E402
    BLUE,
    ORANGE,
    TEAL,
    GOLD,
    PURPLE,
    GREY,
    LIGHT,
    DARK,
    CMAP_SEQ,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "server_results_20260728"
ASSETS = FROZEN / "results" / "manuscript_assets_frozen_20260721" / "tables"
STATS = FROZEN / "results" / "manuscript_statistics_frozen_20260721"
ROBUSTNESS = FROZEN / "results" / "robustness" / "frozen_20260721"
INTERNAL = FROZEN / "results" / "internal_cv" / "lr_cpg_panel_frozen_20260721"
OUT = ROOT / "results" / "manuscript_figures_frozen_revision"
OUT.mkdir(parents=True, exist_ok=True)
GENERATED_TABLE = ROOT / "submission" / "manuscript" / "generated_per_class_table.tex"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    # dpi is passed explicitly so that any raster element embedded in the PDF
    # (imshow panels, colour bars) is written at >=300 dpi, not the 100 dpi default.
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
    )


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.75,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(box)
    ax.add_patch(
        FancyBboxPatch(
            (x, y + height - 0.07),
            width,
            0.07,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + width / 2,
        y + height - 0.035,
        title,
        color="white",
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=6.3,
    )
    ax.text(
        x + width / 2,
        y + (height - 0.07) / 2,
        body,
        ha="center",
        va="center",
        color=DARK,
        fontsize=5.8,
        linespacing=1.35,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.7,
            color=GREY,
        )
    )


def build_figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax,
        (0.035, 0.59),
        0.22,
        0.29,
        "DEVELOPMENT",
        "9,065 tumour samples\n8,923 patients\n33 TCGA classes\n413,341 candidate CpGs",
        BLUE,
    )
    draw_box(
        ax,
        (0.31, 0.59),
        0.25,
        0.29,
        "INTERNAL CV",
        "Patient-stratified 5-fold CV\nrepeated three times\nFold-specific imputation,\nranking, panel and LR model",
        TEAL,
    )
    draw_box(
        ax,
        (0.615, 0.59),
        0.34,
        0.29,
        "PRIMARY INTERNAL ESTIMATE",
        "One OOF probability vector per patient\nwithin each repeat\nMetrics calculated per repeat\nthen averaged; patient bootstrap CI",
        PURPLE,
    )
    arrow(ax, (0.255, 0.735), (0.31, 0.735))
    arrow(ax, (0.56, 0.735), (0.615, 0.735))

    draw_box(
        ax,
        (0.035, 0.13),
        0.25,
        0.27,
        "CONSENSUS PANELS",
        "15 fold-specific rankings\nselected count ↓, mean rank ↑\nLocked 500-CpG consensus panel\nLocked 1,000-CpG consensus panel",
        GOLD,
    )
    draw_box(
        ax,
        (0.36, 0.13),
        0.23,
        0.27,
        "FULL-DEVELOPMENT LR",
        "One LR model per panel\ntrained on all 9,065 tumours\nFull-development imputation,\nscaling and class order",
        ORANGE,
    )
    draw_box(
        ax,
        (0.665, 0.06),
        0.29,
        0.41,
        "EXTERNAL VALIDATION",
        "GSE56044 (106) — unrestricted 33-class\n"
        "GSE53051 (102) — 5 lineages + other\n"
        "GSE48684 (64) — colorectal recall\n"
        "GSE105260 (35) — KIRC recall",
        BLUE,
    )
    arrow(ax, (0.285, 0.265), (0.36, 0.265))
    arrow(ax, (0.59, 0.265), (0.665, 0.265))
    ax.text(
        0.5,
        0.965,
        "Patient-level development and consensus-panel external validation",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )
    save(fig, "figure1_study_design")


def build_figure_2() -> None:
    estimates = pd.read_csv(
        STATS / "internal_patient_oof_primary_estimates_ci.tsv", sep="\t"
    )
    repeats = pd.read_csv(STATS / "internal_patient_oof_metrics_by_repeat.tsv", sep="\t")
    stability = pd.read_csv(ASSETS / "table_panel_stability.tsv", sep="\t")
    delta = pd.read_csv(
        ASSETS / "table_internal_500_vs_1000_paired_bootstrap.tsv", sep="\t"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35))

    ax = axes[0, 0]
    macro = estimates[estimates["metric"] == "macro_f1"].copy()
    x = np.arange(len(macro))
    colors = [BLUE if p == 500 else ORANGE for p in macro["panel_size"]]
    for i, (_, row) in enumerate(macro.iterrows()):
        color = BLUE if row["panel_size"] == 500 else ORANGE
        ax.errorbar(
            i,
            row["estimate_mean_across_repeats"],
            yerr=[
                [row["estimate_mean_across_repeats"] - row["ci_low"]],
                [row["ci_high"] - row["estimate_mean_across_repeats"]],
            ],
            fmt="o",
            color=color,
            capsize=4,
            linewidth=1.5,
            markersize=6,
            zorder=3,
        )
    for repeat in (0, 1, 2):
        vals = repeats[repeats["repeat"] == repeat].sort_values("panel_size")
        ax.plot(
            x,
            vals["macro_f1"],
            color=LIGHT,
            linewidth=0.8,
            marker="o",
            markersize=2.5,
            zorder=1,
        )
    ax.set_xticks(x, [f"{int(p):,}" for p in macro["panel_size"]])
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Patient-level macro-F1")
    ax.set_ylim(0.905, 0.94)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    d = delta.iloc[0]
    ax.text(
        0.03,
        0.06,
        f"1,000−500: {d['estimate']:.4f}\n"
        f"95% CI {d['ci_low']:.4f} to {d['ci_high']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": LIGHT, "edgecolor": LIGHT},
    )
    ax.set_title("Repeat-specific and primary macro-F1", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "a")

    ax = axes[0, 1]
    metric_order = ["accuracy", "balanced_accuracy"]
    labels = ["Accuracy", "Balanced\naccuracy"]
    for panel in (500, 1000):
        sub = estimates[
            (estimates["panel_size"] == panel)
            & estimates["metric"].isin(metric_order)
        ].set_index("metric")
        for i, metric in enumerate(metric_order):
            row = sub.loc[metric]
            offset = -0.08 if panel == 500 else 0.08
            ax.errorbar(
                i + offset,
                row["estimate_mean_across_repeats"],
                yerr=[
                    [row["estimate_mean_across_repeats"] - row["ci_low"]],
                    [row["ci_high"] - row["estimate_mean_across_repeats"]],
                ],
                fmt="o",
                color=BLUE if panel == 500 else ORANGE,
                capsize=3,
                linewidth=1.3,
                markersize=5.5,
                label=f"{panel:,} CpG" if i == 0 else None,
            )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Patient-level estimate")
    ax.set_ylim(0.91, 0.97)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Discrimination metrics", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "b")

    ax = axes[1, 0]
    metric_order = ["brier_multiclass", "ece_15_bins", "aurc"]
    labels = ["Brier score", "ECE", "AURC"]
    for panel in (500, 1000):
        sub = estimates[
            (estimates["panel_size"] == panel)
            & estimates["metric"].isin(metric_order)
        ].set_index("metric")
        vals = [sub.loc[m, "estimate_mean_across_repeats"] for m in metric_order]
        xpos = np.arange(len(metric_order)) + (-0.17 if panel == 500 else 0.17)
        ax.bar(
            xpos,
            vals,
            width=0.32,
            color=BLUE if panel == 500 else ORANGE,
            label=f"{panel:,} CpG",
        )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Metric value (lower is better)")
    ax.set_ylim(0, 0.10)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Probability and selective-classification metrics", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "c")

    ax = axes[1, 1]
    cats = ["Nogueira", "Jaccard", "RBO", "15/15\nfraction", "≥12/15\nfraction"]
    for _, row in stability.iterrows():
        values = [
            row["nogueira_stability"],
            row["pairwise_jaccard_mean"],
            row["rank_biased_overlap_mean"],
            row["selected_15_of_15_count"] / row["panel_size"],
            row["selected_ge_12_of_15_count"] / row["panel_size"],
        ]
        xpos = np.arange(len(cats)) + (-0.17 if row["panel_size"] == 500 else 0.17)
        ax.bar(
            xpos,
            values,
            width=0.32,
            color=BLUE if row["panel_size"] == 500 else ORANGE,
            label=f"{int(row['panel_size']):,} CpG",
        )
    ax.set_xticks(np.arange(len(cats)), cats)
    ax.set_ylabel("Stability or selected proportion")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_title("Panel stability and recurrent selection", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "d")

    fig.tight_layout(w_pad=2.3, h_pad=2.9)
    save(fig, "figure2_internal_stability")


def build_figure_3() -> None:
    ext = pd.read_csv(ASSETS / "table_external_frozen_endpoints_ci.tsv", sep="\t")
    ext["label"] = (
        ext["cohort"]
        + " — "
        + ext["panel_size"].map(lambda x: f"{int(x):,} CpG")
    )
    cohort_order = ["GSE56044", "GSE53051", "GSE48684", "GSE105260"]
    ext["cohort_rank"] = ext["cohort"].map({c: i for i, c in enumerate(cohort_order)})
    ext = ext.sort_values(["cohort_rank", "panel_size"], ascending=[False, False]).reset_index(drop=True)

    fig = plt.figure(figsize=(7.15, 4.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(ext))
    colors = [BLUE if p == 500 else ORANGE for p in ext["panel_size"]]
    for i, (_, row) in enumerate(ext.iterrows()):
        color = BLUE if row["panel_size"] == 500 else ORANGE
        ax.errorbar(
            row["estimate"],
            i,
            xerr=[
                [row["estimate"] - row["exact_binomial_95ci_low"]],
                [row["exact_binomial_95ci_high"] - row["estimate"]],
            ],
            fmt="o",
            color=color,
            capsize=3,
            linewidth=1.5,
            markersize=5.5,
            zorder=3,
        )
    ax.set_yticks(y, ext["label"])
    ax.set_xlim(0.60, 1.01)
    ax.set_xlabel("External endpoint estimate (exact 95% binomial CI)")
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.set_title("Consensus-panel external validation", loc="left", fontweight="bold")
    for i, (_, row) in enumerate(ext.iterrows()):
        if row["exact_binomial_95ci_high"] > 0.95:
            text_x = row["exact_binomial_95ci_low"] - 0.012
            align = "right"
        else:
            text_x = row["exact_binomial_95ci_high"] + 0.012
            align = "left"
        ax.text(
            text_x,
            i,
            f"{int(row['successes'])}/{int(row['n_samples'])}",
            va="center",
            ha=align,
            fontsize=7.2,
            color=DARK,
        )
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    endpoint_text = [
        ("GSE56044", "Unrestricted 33-class\ntop-1 prediction", BLUE),
        ("GSE53051", "Five lineages plus other\nsix-state prediction", TEAL),
        ("GSE48684", "Positive-class colorectal\nlineage recall", GOLD),
        ("GSE105260", "Positive-class KIRC\nsingle-class recall", ORANGE),
    ]
    ys = [0.82, 0.62, 0.42, 0.22]
    for (name, desc, color), yy in zip(endpoint_text, ys):
        ax.add_patch(
            FancyBboxPatch(
                (0.02, yy - 0.075),
                0.96,
                0.15,
                boxstyle="round,pad=0.015",
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.08,
            yy + 0.025,
            name,
            transform=ax.transAxes,
            fontweight="bold",
            color=color,
            va="center",
        )
        ax.text(0.08, yy - 0.035, desc, transform=ax.transAxes, va="center", fontsize=8)
    ax.text(
        0.02,
        0.98,
        "Endpoint definitions",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.02,
        0.03,
        "Positive-only cohorts do not estimate\nspecificity, balanced accuracy or full calibration.",
        transform=ax.transAxes,
        fontsize=7.8,
        color=GREY,
        va="bottom",
    )
    panel_label(ax, "b")
    save(fig, "figure3_external_validation")


def build_figure_4() -> None:
    summaries = []
    effects = []
    ood_rows = []
    for panel in (500, 1000):
        panel_dir = ROBUSTNESS / f"consensus_locked_{panel}_cpg"

        summary = pd.read_csv(panel_dir / "robustness_seed_summary.tsv", sep="\t")
        summary["panel_size"] = panel
        summaries.append(summary)

        effect = pd.read_csv(
            panel_dir / "robustness_confirmatory_effects_bootstrap_ci.tsv",
            sep="\t",
        )
        effect["panel_size"] = panel
        effects.append(effect)

        ood = pd.read_csv(panel_dir / "normal_ood_summary.tsv", sep="\t")
        ood["panel_size"] = panel
        ood_rows.append(ood)

    summary = pd.concat(summaries, ignore_index=True)
    effects = pd.concat(effects, ignore_index=True)
    ood = pd.concat(ood_rows, ignore_index=True)

    fig = plt.figure(figsize=(7.15, 5.45))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 1.0],
        hspace=0.42,
        wspace=0.34,
    )
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    ax = axes[0]
    admixture = summary[
        (summary["condition"] == "added_normal_profile_fraction")
        & (summary["metric"] == "macro_f1")
    ].copy()
    for panel, color in ((500, BLUE), (1000, ORANGE)):
        sub = admixture[admixture["panel_size"] == panel].sort_values("level")
        x = sub["level"].to_numpy(dtype=float)
        median = sub["seed_median"].to_numpy(dtype=float)
        lower = sub["seed_q_0_025"].to_numpy(dtype=float)
        upper = sub["seed_q_0_975"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            x,
            median,
            color=color,
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            label=f"{panel:,} CpG",
        )
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0.58, 1.02)
    ax.set_xlabel("Added normal-profile fraction")
    ax.set_ylabel("Seed-median macro-F1")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Same-project normal admixture", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "a")

    ax = axes[1]
    condition_order = [
        "added_normal_profile_fraction",
        "uniform_panel_probe_loss",
    ]
    condition_labels = ["40% added\nnormal profile", "20% uniform\nprobe loss"]
    x = np.arange(len(condition_order))
    for panel, color, offset in ((500, BLUE, -0.18), (1000, ORANGE, 0.18)):
        sub = (
            effects[effects["panel_size"] == panel]
            .set_index("condition")
            .loc[condition_order]
        )
        values = sub["estimate_delta"].to_numpy(dtype=float)
        lower = sub["ci_low"].to_numpy(dtype=float)
        upper = sub["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([values - lower, upper - values])
        ax.bar(
            x + offset,
            values,
            width=0.32,
            color=color,
            label=f"{panel:,} CpG",
            zorder=2,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=yerr,
            fmt="none",
            ecolor=DARK,
            elinewidth=1,
            capsize=2.5,
            zorder=3,
        )
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_xticks(x, condition_labels)
    ax.set_ylim(-0.16, 0.01)
    ax.set_ylabel(r"$\Delta$macro-F1 vs clean input")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_title("Prespecified confirmatory conditions", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "b")

    ax = axes[2]
    metrics = [
        "cancer_class_false_acceptance_at_confidence_threshold",
        "organ_lineage_concordance",
        "mean_max_probability",
    ]
    labels = [
        "High-confidence\ntumour-class\nassignment",
        "Organ-lineage\nconcordance",
        "Mean maximum\nconfidence",
    ]
    x = np.arange(3)
    for _, row in ood.iterrows():
        offset = -0.17 if row["panel_size"] == 500 else 0.17
        ax.bar(
            x + offset,
            [row[m] for m in metrics],
            width=0.32,
            color=BLUE if row["panel_size"] == 500 else ORANGE,
            label=f"{int(row['panel_size']):,} CpG",
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion or mean score")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
    )
    ax.set_title("Normal-tissue stress test (n=747)", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.10)
    save(fig, "figure4_robustness_ood")


def derive_class_metrics() -> tuple[pd.DataFrame, dict[int, np.ndarray], list[str]]:
    raw = pd.read_csv(
        INTERNAL / "repeated_cross_fitting_ensemble_confusion_matrices.tsv",
        sep="\t",
    )
    labels = [col for col in raw.columns if col.startswith("TCGA-")]
    rows = []
    matrices: dict[int, np.ndarray] = {}
    for panel in (500, 1000):
        sub = raw[raw["panel_size"] == panel].set_index("true_label").loc[labels, labels]
        matrix = sub.to_numpy(dtype=float)
        matrices[panel] = matrix
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        tp = np.diag(matrix)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        for idx, label in enumerate(labels):
            errors = matrix[idx].copy()
            errors[idx] = 0
            top_idx = int(np.argmax(errors))
            top_count = int(errors[top_idx])
            rows.append(
                {
                    "panel_size": panel,
                    "class": label.replace("TCGA-", ""),
                    "n_patients": int(support[idx]),
                    "recall": recall[idx],
                    "precision": precision[idx],
                    "f1": f1[idx],
                    "top_confused_class": (
                        labels[top_idx].replace("TCGA-", "") if top_count else "None"
                    ),
                    "top_confused_count": top_count,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "internal_class_level_metrics.tsv", sep="\t", index=False)
    return metrics, matrices, labels


def write_class_table(metrics: pd.DataFrame) -> None:
    wide = metrics.pivot(index="class", columns="panel_size")
    classes = metrics[metrics["panel_size"] == 1000].sort_values("class")["class"]
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{longtable}{@{}lrrrrrl@{}}",
        r"\caption{Class-level descriptive performance of the repeated-cross-fitting ensemble. These values were calculated from probabilities averaged across the three repeats and are provided for error localization; they are not the primary repeat-specific performance estimator.}\label{tab:per-class}\\",
        r"\toprule",
        r"Class & $n$ & Recall, 500 & F1, 500 & Recall, 1,000 & F1, 1,000 & Dominant 1,000-CpG error ($n$) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Class & $n$ & Recall, 500 & F1, 500 & Recall, 1,000 & F1, 1,000 & Dominant 1,000-CpG error ($n$) \\",
        r"\midrule",
        r"\endhead",
    ]
    for cls in classes:
        n = int(wide.loc[cls, ("n_patients", 1000)])
        r500 = wide.loc[cls, ("recall", 500)]
        f500 = wide.loc[cls, ("f1", 500)]
        r1000 = wide.loc[cls, ("recall", 1000)]
        f1000 = wide.loc[cls, ("f1", 1000)]
        confused = wide.loc[cls, ("top_confused_class", 1000)]
        confused_n = int(wide.loc[cls, ("top_confused_count", 1000)])
        error_text = "None" if confused_n == 0 else f"{confused} ({confused_n})"
        lines.append(
            f"{cls} & {n} & {r500:.3f} & {f500:.3f} & "
            f"{r1000:.3f} & {f1000:.3f} & {error_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    GENERATED_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figure_5() -> None:
    metrics, matrices, labels = derive_class_metrics()
    write_class_table(metrics)
    short_labels = [label.replace("TCGA-", "") for label in labels]
    matrix = matrices[1000]
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 5.75),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
    )

    ax = axes[0]
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap=CMAP_SEQ, aspect="auto")
    ax.set_xticks(np.arange(len(short_labels)), short_labels, rotation=90)
    ax.set_yticks(np.arange(len(short_labels)), short_labels)
    ax.set_xlabel("Predicted TCGA class")
    ax.set_ylabel("True TCGA class")
    ax.set_title("1,000-CpG normalized confusion matrix", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "a")
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Within-class proportion")

    ax = axes[1]
    wide = metrics.pivot(index="class", columns="panel_size")
    order = wide[("f1", 1000)].sort_values().index
    y = np.arange(len(order))
    ax.scatter(wide.loc[order, ("f1", 500)], y, color=BLUE, s=22, label="500 CpG")
    ax.scatter(wide.loc[order, ("f1", 1000)], y, color=ORANGE, s=22, label="1,000 CpG")
    for yy, cls in zip(y, order):
        ax.plot(
            [wide.loc[cls, ("f1", 500)], wide.loc[cls, ("f1", 1000)]],
            [yy, yy],
            color=LIGHT,
            linewidth=0.8,
            zorder=0,
        )
    ax.set_yticks(y, [f"{cls} (n={int(wide.loc[cls, ('n_patients', 1000)])})" for cls in order])
    ax.set_xlim(0.35, 1.01)
    ax.set_xlabel("Descriptive class F1")
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Class F1 and patient count", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "b")

    fig.tight_layout(w_pad=1.6)
    save(fig, "figure5_class_performance")


def build_supp_figure_reactome() -> None:
    reactome = pd.read_csv(ASSETS / "table_reactome_top50_500_cpg.tsv", sep="\t")
    reactome = reactome.sort_values("standardized_enrichment_effect", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y = np.arange(len(reactome))[::-1]
    ax.barh(y, reactome["standardized_enrichment_effect"], color=BLUE)
    shortened = [
        s.replace("Antigen processing: Ubiquitination & Proteasome degradation", "Antigen processing / proteasome")
        .replace("NOTCH4 Intracellular Domain Regulates Transcription", "NOTCH4 domain transcription")
        .replace("Mitotic G1 phase and G1/S transition", "Mitotic G1 and G1/S")
        .replace("Transcriptional regulation by RUNX3", "RUNX3 transcriptional regulation")
        for s in reactome["pathway"]
    ]
    ax.set_yticks(y, shortened)
    ax.set_xlabel("Standardized matched-null enrichment effect")
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.set_title("Primary 500-CpG Reactome analysis", loc="left", fontweight="bold", fontsize=9)
    ax.text(
        0.98,
        0.02,
        "No pathway passed BH FDR 0.05",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GREY,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": LIGHT, "linewidth": 0.5, "pad": 1.8, "alpha": 0.95},
    )
    fig.tight_layout()
    save(fig, "supp_figure2_reactome")


def build_supp_figure_pathmethnet() -> None:
    metric_files = [
        ROOT
        / "results"
        / "internal_cv"
        / "model_artifacts"
        / "q1_pathmethnet_v2_gpu0_runs01_08"
        / "fold_metrics.tsv",
        ROOT
        / "results"
        / "internal_cv"
        / "model_artifacts"
        / "q1_pathmethnet_v2_gpu1_runs09_15"
        / "fold_metrics.tsv",
    ]
    folds = pd.concat(
        [pd.read_csv(path, sep="\t") for path in metric_files],
        ignore_index=True,
    ).sort_values(["repeat", "fold"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.8),
        gridspec_kw={"width_ratios": [1.45, 0.75]},
    )
    rng = np.random.default_rng(20260728)

    ax = axes[0]
    metrics = ["macro_f1", "balanced_accuracy", "accuracy"]
    labels = ["Macro-F1", "Balanced\naccuracy", "Accuracy"]
    for idx, metric in enumerate(metrics):
        values = folds[metric].to_numpy(dtype=float)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            color=BLUE,
            alpha=0.75,
            s=22,
        )
        ax.boxplot(
            values,
            positions=[idx],
            widths=0.35,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": LIGHT, "edgecolor": DARK},
            medianprops={"color": ORANGE, "linewidth": 1.6},
            whiskerprops={"color": DARK},
            capprops={"color": DARK},
        )
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylim(0.83, 0.94)
    ax.set_ylabel("Held-out fold metric")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_title("Discrimination across 15 folds", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "A")

    ax = axes[1]
    values = folds["log_loss"].to_numpy(dtype=float)
    jitter = rng.normal(0, 0.035, size=len(values))
    ax.scatter(jitter, values, color=ORANGE, alpha=0.75, s=24)
    ax.boxplot(
        values,
        positions=[0],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": LIGHT, "edgecolor": DARK},
        medianprops={"color": BLUE, "linewidth": 1.6},
        whiskerprops={"color": DARK},
        capprops={"color": DARK},
    )
    ax.set_xticks([0], ["Log loss"])
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylabel("Held-out fold log loss")
    ax.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax.set_title("Probability loss", loc="left", fontweight="bold", fontsize=9)
    panel_label(ax, "B")

    fig.suptitle(
        "Exploratory cross-validation performance of PathMethNet",
        fontsize=10,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(w_pad=2.0)
    save(fig, "supp_figure1_pathmethnet")


if __name__ == "__main__":
    # Submission entry point: use the revised data-forward figure set.  The
    # legacy builders remain above for traceability, but are not manuscript
    # inputs because they use presentation-style layouts.
    from build_submission_final_figures import main
    main()
