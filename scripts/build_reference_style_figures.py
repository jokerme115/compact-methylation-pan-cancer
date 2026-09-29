#!/usr/bin/env python3
"""Build compact, data-forward figures using the visual grammar of the reference papers."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/manuscript_figures_reference_style_v2"
SUPP_OUT = OUT / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
SUPP_OUT.mkdir(parents=True, exist_ok=True)

BLACK = "#202020"
GRAY = "#777777"
LIGHT = "#D9D9D9"
BLUE = "#2166AC"
RED = "#B2182B"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, stem: str, supplementary: bool = False) -> None:
    destination = SUPP_OUT if supplementary else OUT
    fig.savefig(destination / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(destination / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure1() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), gridspec_kw={"width_ratios": [0.95, 1.35, 1.05]})
    for ax in axes:
        ax.set_axis_off()

    ax = axes[0]
    panel_label(ax, "a")
    ax.set_title("Development cohort", loc="left", fontweight="bold", pad=8, fontsize=7.5)
    stages = [(0.08, 0.72, "TCGA 450K\n9,065 tumours"), (0.08, 0.43, "33 cancer\nclasses"), (0.08, 0.14, "Locked patient\nidentifiers")]
    for x, y, text in stages:
        ax.add_patch(Rectangle((x, y), 0.72, 0.18, fill=False, ec=BLACK, lw=0.8))
        ax.text(x + 0.36, y + 0.09, text, ha="center", va="center")
    for y1, y2 in [(0.72, 0.61), (0.43, 0.32)]:
        ax.add_patch(FancyArrowPatch((0.44, y1), (0.44, y2), arrowstyle="-|>", mutation_scale=8, lw=0.7, color=BLACK))

    ax = axes[1]
    panel_label(ax, "b")
    ax.set_title("Leakage-free model development", loc="left", fontweight="bold", pad=8, fontsize=7.5)
    items = [
        (0.02, 0.66, "Patient-level\n5-fold × 3 CV"),
        (0.57, 0.66, "Training-fold\nfeature ranking"),
        (0.02, 0.25, "LR panels\n10–1,000 CpGs"),
        (0.57, 0.25, "PathMethNet\nCpG→gene→pathway"),
    ]
    for x, y, text in items:
        ax.add_patch(Rectangle((x, y), 0.38, 0.2, facecolor="white", edgecolor=BLACK, lw=0.8))
        ax.text(x + 0.19, y + 0.1, text, ha="center", va="center", fontsize=6.5)
    for start, end in [((0.40, 0.76), (0.57, 0.76)), ((0.21, 0.66), (0.21, 0.45)), ((0.76, 0.66), (0.76, 0.45))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, lw=0.7, color=BLACK))
    ax.text(0.49, 0.05, "All preprocessing fitted without held-out samples", ha="center", color=RED, fontsize=6.6)

    ax = axes[2]
    panel_label(ax, "c")
    ax.set_title("Independent validation", loc="left", fontweight="bold", pad=8, fontsize=7.5)
    cohorts = [("GSE56044", "n=106", "LUAD/LUSC"), ("GSE48684", "n=64", "colorectal"), ("GSE53051", "n=102", "5 lineages"), ("GSE105260", "n=35", "KIRC")]
    y = 0.78
    for name, n, label in cohorts:
        ax.plot([0.08, 0.22], [y, y], color=BLUE, lw=2.4, solid_capstyle="butt")
        ax.text(0.26, y + 0.025, name, fontweight="bold", va="center")
        ax.text(0.26, y - 0.055, f"{n}; {label}", color=GRAY, va="center")
        y -= 0.2
    ax.text(0.08, 0.02, "Exact histology and accepted-lineage\nevaluations reported separately", fontsize=6.6)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig1_study_design_reference_style")


def figure2() -> None:
    folds = pd.read_csv(ROOT / "results/internal_cv/lr_cpg_panel/fold_metrics.tsv", sep="\t")
    summary = pd.read_csv(ROOT / "results/manuscript_statistics/internal_cv_bootstrap_ci.tsv", sep="\t")
    panels = [10, 20, 50, 100, 200, 500, 1000]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    ax = axes[0]
    panel_label(ax, "a")
    for metric, color, label in [("macro_f1", BLACK, "Macro F1"), ("accuracy", BLUE, "Accuracy"), ("top3_accuracy", GRAY, "Top-3 accuracy")]:
        agg = folds.groupby("panel_size")[metric].agg(["mean", "std"]).reindex(panels)
        ax.errorbar(panels, agg["mean"], yerr=agg["std"], marker="o", ms=3.2, lw=1.1, capsize=2, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(panels, [str(x) for x in panels], rotation=45)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Held-out performance")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)

    ax = axes[1]
    panel_label(ax, "b")
    focus = folds[folds.panel_size.isin([100, 200, 500, 1000])]
    positions = np.arange(4)
    rng = np.random.default_rng(4)
    for i, panel in enumerate([100, 200, 500, 1000]):
        vals = focus.loc[focus.panel_size == panel, "macro_f1"].to_numpy()
        ax.scatter(np.full(vals.size, i) + rng.normal(0, 0.045, vals.size), vals, s=9, facecolors="white", edgecolors=BLACK, lw=0.55, zorder=2)
        ax.plot([i - 0.16, i + 0.16], [vals.mean(), vals.mean()], color=RED, lw=1.2)
    ax.set_xticks(positions, ["100", "200", "500", "1,000"])
    ax.set_ylim(0.81, 0.96)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Macro F1 across 15 folds")
    clean(ax)

    ax = axes[2]
    panel_label(ax, "c")
    for metric, color, label in [("coverage_at_0_90", BLACK, "Coverage ≥0.90"), ("accuracy_at_0_90", BLUE, "Accuracy at ≥0.90")]:
        agg = folds.groupby("panel_size")[metric].mean().reindex(panels)
        ax.plot(panels, agg, marker="o", ms=3.2, lw=1.1, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(panels, [str(x) for x in panels], rotation=45)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Proportion")
    ax.legend(frameon=False, loc="lower right")
    retained = folds[folds.panel_size.isin([100, 200, 500, 1000])].assign(
        retained_n=lambda d: (d.coverage_at_0_90 * d.n_test).round()
    ).groupby("panel_size").retained_n.median().reindex([100, 200, 500, 1000])
    ax.text(0.03, 0.04, "Median retained n at ≥0.90: " + ", ".join(
        f"{int(v)} ({p})" for p, v in retained.items()
    ), transform=ax.transAxes, fontsize=5.8, color=GRAY)
    clean(ax)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig2_internal_performance_reference_style")


def figure3() -> None:
    stats = pd.read_csv(ROOT / "results/manuscript_statistics/external_validation_confidence_intervals.tsv", sep="\t")
    labels = {"GSE56044": "GSE56044\nLUAD/LUSC", "GSE48684": "GSE48684\ncolorectal", "GSE53051": "GSE53051\n5 lineages", "GSE105260_official_series": "GSE105260\nKIRC"}
    short_labels = {"GSE56044": "GSE56044\nexact", "GSE48684": "GSE48684\naccepted", "GSE53051": "GSE53051\naccepted", "GSE105260_official_series": "GSE105260\nexact"}
    endpoint_types = {"GSE56044": "Exact", "GSE48684": "Accepted lineage", "GSE53051": "Accepted lineage", "GSE105260_official_series": "Exact"}
    cohorts = list(labels)
    panels = [100, 200, 500, 1000]
    matrix = stats.pivot(index="cohort", columns="panel_size", values="accepted_top1").reindex(cohorts)[panels]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.5), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})

    ax = axes[0]
    panel_label(ax, "a")
    im = ax.imshow(matrix.to_numpy(), vmin=0.6, vmax=1.0, cmap="Greys", aspect="auto")
    for i in range(len(cohorts)):
        for j in range(len(panels)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if value >= 0.82 else BLACK, fontsize=6.8)
    ax.set_xticks(range(4), ["100", "200", "500", "1,000"])
    ax.set_yticks(range(4), [short_labels[x] for x in cohorts])
    ax.set_xlabel("CpG panel size")
    ax.set_title("Top-1 accuracy\n(exact or accepted lineage, as defined)", pad=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    panel_label(ax, "b")
    anchor = stats[stats.panel_size == 1000].set_index("cohort").loc[cohorts]
    y = np.arange(len(cohorts))[::-1]
    x = anchor.accepted_top1.to_numpy()
    lo = anchor.accepted_top1_95ci_low.to_numpy()
    hi = anchor.accepted_top1_95ci_high.to_numpy()
    markers = {"Exact": "o", "Accepted lineage": "s"}
    for idx, cohort in enumerate(cohorts):
        marker = markers[endpoint_types[cohort]]
        ax.errorbar(x[idx], y[idx], xerr=[[x[idx] - lo[idx]], [hi[idx] - x[idx]]], fmt=marker,
                    color=BLACK, ecolor=BLACK, capsize=2, ms=4,
                    label=endpoint_types[cohort] if endpoint_types[cohort] not in [endpoint_types[c] for c in cohorts[:idx]] else None)
    ax.axvline(0.8, color=LIGHT, lw=0.8, ls="--")
    ax.set_yticks(y, [short_labels[c] for c in cohorts])
    ax.set_xlim(0.45, 1.01)
    ax.set_xlabel("1,000-CpG top-1 accuracy (95% CI)")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)
    clean(ax)

    ax = axes[2]
    panel_label(ax, "c")
    kidney = stats[stats.cohort == "GSE105260_official_series"].set_index("panel_size").loc[panels]
    x = np.arange(4)
    vals = kidney.accepted_top1.to_numpy()
    lo = kidney.accepted_top1_95ci_low.to_numpy()
    hi = kidney.accepted_top1_95ci_high.to_numpy()
    ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="o-", color=BLACK, capsize=2, ms=4, lw=1)
    ax.set_xticks(x, ["100", "200", "500", "1,000"])
    ax.set_ylim(0.4, 1.01)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("GSE105260 exact KIRC accuracy")
    ax.set_title("Exploratory panel-size behavior", pad=6)
    clean(ax)
    fig.subplots_adjust(wspace=0.5)
    save(fig, "fig3_external_validation_reference_style")


def figure4() -> None:
    stable = pd.read_csv(ROOT / "results/internal_cv/lr_cpg_panel/stable_cpg_panel_annotated.tsv", sep="\t")
    classes = pd.read_csv(ROOT / "results/internal_cv/lr_cpg_panel/class_specific_top_cpg_annotated.tsv", sep="\t")
    top = stable.sort_values(["selected_count", "mean_rank"], ascending=[False, True]).head(100).copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.1))

    ax = axes[0, 0]
    panel_label(ax, "a")
    context = pd.Series({
        "Promoter": top.is_promoter.fillna(False).astype(bool).sum(),
        "Island": top.relation_to_cpg_island.fillna("").eq("Island").sum(),
        "Shore": top.relation_to_cpg_island.fillna("").str.contains("Shore").sum(),
        "Shelf": top.relation_to_cpg_island.fillna("").str.contains("Shelf").sum(),
        "Gene annotated": top.gene_symbol.fillna("").ne("").sum(),
    }).sort_values()
    ax.barh(context.index, context.values, color=GRAY)
    for y, value in enumerate(context.values):
        ax.text(value + 1, y, str(value), va="center", fontsize=7)
    ax.set_xlim(0, 105)
    ax.set_xlabel("CpGs among stable top 100")
    clean(ax)

    ax = axes[0, 1]
    panel_label(ax, "b")
    genes = classes[classes.gene_symbol.fillna("").ne("")].groupby("gene_symbol").agg(mean_abs_coef=("mean_abs_coef", "mean"), classes=("label", "nunique"), folds=("selected_fold_count", "mean")).sort_values("mean_abs_coef", ascending=False).head(15).sort_values("mean_abs_coef")
    ax.barh(genes.index, genes.mean_abs_coef, color=BLACK)
    ax.set_xlabel("Mean absolute class coefficient")
    clean(ax)

    ax = axes[1, 0]
    panel_label(ax, "c")
    annotation = pd.read_csv(ROOT / "data/annotations/illumina450k_probe_gene.tsv", sep="\t")
    reactome = pd.read_csv(ROOT / "data/annotations/reactome_pathway_gene.tsv", sep="\t")
    selected = stable[stable.panel_size.eq(1000)][["probe_id"]].drop_duplicates()
    mapped = selected.merge(annotation, on="probe_id", how="left").dropna(subset=["gene_symbol"])
    mapped = mapped.merge(reactome, on="gene_symbol", how="inner")
    pathways = mapped["pathway"].value_counts().head(12).sort_values()
    wrapped = [textwrap.fill(x, 34) for x in pathways.index]
    ax.barh(wrapped, pathways.values, color=BLUE)
    ax.set_xlabel("Mapped panel CpGs (count; not an enrichment score)")
    ax.set_title("Annotation-supported pathway links", pad=5)
    ax.text(0.98, 0.02, f"{mapped.probe_id.nunique()} CpGs; {mapped.gene_symbol.nunique()} genes; no enrichment test",
            transform=ax.transAxes, ha="right", fontsize=5.8, color=GRAY)
    clean(ax)

    ax = axes[1, 1]
    panel_label(ax, "d")
    selected_classes = ["TCGA-BRCA", "TCGA-COAD", "TCGA-READ", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LUAD", "TCGA-LUSC", "TCGA-PAAD", "TCGA-THCA"]
    subset = classes[classes.label.isin(selected_classes) & classes.gene_symbol.fillna("").ne("")].copy()
    top_genes = subset.groupby("gene_symbol").mean_abs_coef.mean().nlargest(18).index
    heat = subset[subset.gene_symbol.isin(top_genes)].pivot_table(index="gene_symbol", columns="label", values="mean_coef", aggfunc="mean", fill_value=0).reindex(index=top_genes, columns=selected_classes).fillna(0)
    vmax = np.quantile(np.abs(heat.to_numpy()), 0.95) or 1
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(selected_classes)), [x.replace("TCGA-", "") for x in selected_classes], rotation=45, ha="right")
    ax.set_yticks(range(len(top_genes)), top_genes)
    ax.set_xlabel("Cancer class")
    ax.set_title("Class-specific LR coefficients", pad=5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Mean coefficient")
    fig.subplots_adjust(hspace=0.42, wspace=0.55)
    save(fig, "fig4_biomarker_interpretation_reference_style")


def figure5() -> None:
    """Confidence/coverage behavior, following the threshold figures in the CNS reference paper."""
    folds = pd.read_csv(ROOT / "results/internal_cv/lr_cpg_panel/fold_metrics.tsv", sep="\t")
    panels = [100, 200, 500, 1000]
    thresholds = np.arange(0.50, 1.00, 0.05)
    # Fold summaries contain only the 0.90 operating point; use the available
    # confidence summaries as a conservative panel-size comparison and show
    # the external sample-level threshold curves in the second row.
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), gridspec_kw={"width_ratios": [1.05, 1.05, 1.2]})
    ax = axes[0]; panel_label(ax, "a")
    agg = folds[folds.panel_size.isin(panels)].groupby("panel_size")
    cov = agg.coverage_at_0_90.mean().reindex(panels)
    acc = agg.accuracy_at_0_90.mean().reindex(panels)
    ax.plot(panels, cov, "o-", color=GRAY, label="Coverage")
    ax.plot(panels, acc, "o-", color=BLUE, label="Accuracy among calls")
    ax.set_xscale("log"); ax.set_xticks(panels, ["100", "200", "500", "1,000"], rotation=45)
    ax.set_ylim(0, 1.02); ax.set_xlabel("CpG panel size"); ax.set_ylabel("Proportion")
    ax.legend(frameon=False); clean(ax)

    ax = axes[1]; panel_label(ax, "b")
    for cohort, color in [("GSE56044", BLUE), ("GSE53051", RED)]:
        pred = pd.read_csv(ROOT / f"results/external_validation/{cohort}/lr_panel/external_predictions.tsv", sep="\t")
        pred = pred[pred.panel_size == 1000]
        rows = []
        for t in thresholds:
            keep = pred.confidence >= t
            rows.append((t, keep.mean(), pred.loc[keep, "accepted_correct"].astype(bool).mean() if keep.any() else np.nan))
        curve = pd.DataFrame(rows, columns=["threshold", "coverage", "accuracy"])
        ax.plot(curve.threshold, curve.accuracy, color=color, lw=1.2, label=f"{cohort} accuracy")
    ax.set_xlim(0.5, 0.96); ax.set_ylim(0.4, 1.02); ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Accuracy among retained calls")
    ax.legend(frameon=False, fontsize=6.2); clean(ax)

    ax = axes[2]; panel_label(ax, "c")
    for cohort, color in [("GSE56044", BLUE), ("GSE53051", RED)]:
        pred = pd.read_csv(ROOT / f"results/external_validation/{cohort}/lr_panel/external_predictions.tsv", sep="\t")
        pred = pred[pred.panel_size == 1000]
        coverage = [(pred.confidence >= t).mean() for t in thresholds]
        ax.plot(thresholds, coverage, color=color, lw=1.2, label=f"{cohort} coverage")
    ax.set_xlim(0.5, 0.96); ax.set_ylim(0, 1.02); ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Retained sample fraction")
    ax.legend(frameon=False, fontsize=6.2); clean(ax)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig5_confidence_coverage_reference_style", supplementary=True)


def figure6() -> None:
    """Direct confusion displays, following the reference papers' error-structure panels."""
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, cohort, title, labels in [
        (axes[0], "GSE56044", "Exact LUAD/LUSC", ["TCGA-LUAD", "TCGA-LUSC"]),
        (axes[1], "GSE53051", "Accepted five-lineage calls", ["TCGA-BRCA", "TCGA-COADREAD", "TCGA-LUNG", "TCGA-PAAD", "TCGA-THCA"]),
    ]:
        panel_label(ax, "a" if ax is axes[0] else "b")
        pred = pd.read_csv(ROOT / f"results/external_validation/{cohort}/lr_panel/external_predictions.tsv", sep="\t")
        pred = pred[pred.panel_size == 1000].copy()
        if cohort == "GSE53051":
            true = pred.label.fillna("").map(lambda x: "TCGA-COADREAD" if "COADREAD" in x else ("TCGA-LUNG" if x == "TCGA-LUNG" else x))
            pred_class = pred.pred_label.fillna("").map(lambda x: "TCGA-COADREAD" if x in {"TCGA-COAD", "TCGA-READ"} else ("TCGA-LUNG" if x in {"TCGA-LUAD", "TCGA-LUSC"} else x))
        else:
            true = pred.label
            pred_class = pred.pred_label
        mat = pd.crosstab(pd.Categorical(true, categories=labels), pd.Categorical(pred_class, categories=labels)).fillna(0)
        row_sums = mat.sum(axis=1).replace(0, 1)
        norm = mat.div(row_sums, axis=0)
        im = ax.imshow(norm.to_numpy(), vmin=0, vmax=1, cmap="Greys", aspect="auto")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                value = norm.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if value >= 0.55 else BLACK)
        ax.set_xticks(range(len(labels)), [x.replace("TCGA-", "") for x in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), [x.replace("TCGA-", "") for x in labels])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Reference"); ax.set_title(title, fontsize=8, pad=5)
    fig.subplots_adjust(wspace=0.62)
    save(fig, "fig6_external_confusion_reference_style", supplementary=True)


def main() -> None:
    style()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()


if __name__ == "__main__":
    main()
