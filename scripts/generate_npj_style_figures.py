from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "manuscript_figures_npj"
TABLE_DIR = RESULTS / "manuscript_tables"

PALETTE = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "blue": "#2B6CB0",
    "blue_light": "#D8E7F6",
    "teal": "#2A9D8F",
    "teal_light": "#D9F0EC",
    "orange": "#D9902F",
    "orange_light": "#F7E4C3",
    "rose": "#B65B6A",
    "rose_light": "#F1D6DC",
    "purple": "#7B5EA7",
    "purple_light": "#E5DDF3",
    "gray": "#8A8F98",
    "gray_light": "#F3F4F6",
}

COHORTS = {
    "GSE56044": {
        "label": "GSE56044\nLUAD/LUSC",
        "metric": "exact_accuracy",
        "top3": "exact_top3_accuracy",
        "dir": RESULTS / "external_validation/GSE56044",
        "color": PALETTE["blue"],
    },
    "GSE48684": {
        "label": "GSE48684\nCOAD/READ",
        "metric": "accepted_top1_accuracy",
        "top3": "accepted_top3_accuracy",
        "dir": RESULTS / "external_validation/GSE48684",
        "color": PALETTE["teal"],
    },
    "GSE53051": {
        "label": "GSE53051\n5 lineages",
        "metric": "accepted_top1_accuracy",
        "top3": "accepted_top3_accuracy",
        "dir": RESULTS / "external_validation/GSE53051",
        "color": PALETTE["orange"],
    },
    "GSE105260 official": {
        "label": "GSE105260\nKIRC",
        "metric": "exact_accuracy",
        "top3": "exact_top3_accuracy",
        "dir": RESULTS / "external_validation/GSE105260_official_series",
        "color": PALETTE["purple"],
    },
}


def set_style() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def clean_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=2.8, color=PALETTE["ink"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.55)
        ax.set_axisbelow(True)


def load_external() -> pd.DataFrame:
    rows = []
    for cohort, cfg in COHORTS.items():
        metrics = pd.read_csv(cfg["dir"] / "lr_panel/external_panel_metrics.tsv", sep="\t")
        manifest = json.loads((cfg["dir"] / "dataset_manifest.json").read_text())
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "cohort": cohort,
                    "label": cfg["label"],
                    "panel_size": int(row["panel_size"]),
                    "top1": float(row[cfg["metric"]]),
                    "top3": float(row[cfg["top3"]]),
                    "confidence": float(row["mean_confidence"]),
                    "samples": int(row["n_external_samples"]),
                    "label_counts": manifest.get("label_counts", {}),
                    "color": cfg["color"],
                }
            )
    return pd.DataFrame(rows)


def fig1_study_design() -> None:
    samples = pd.read_csv(ROOT / "data/tcga_450k/samples_locked.tsv", sep="\t")
    counts = samples["project_id"].value_counts()
    external = load_external()
    cohort_rows = []
    for cohort, cfg in COHORTS.items():
        manifest = json.loads((cfg["dir"] / "dataset_manifest.json").read_text())
        cohort_rows.append((cohort.replace(" official", ""), int(manifest["n_kept_samples"]), cfg["color"]))

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.18, 1], width_ratios=[1.18, 1], hspace=0.38, wspace=0.28)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_tcga = fig.add_subplot(gs[1, 0])
    ax_ext = fig.add_subplot(gs[1, 1])

    ax_flow.set_axis_off()
    steps = [
        ("TCGA 450k\n9,065 tumors\n33 classes", PALETTE["blue"], PALETTE["blue_light"]),
        ("Patient-level\n5-fold x 3 CV", PALETTE["teal"], PALETTE["teal_light"]),
        ("Fold-wise\nCpG selection", PALETTE["orange"], PALETTE["orange_light"]),
        ("Compact LR\nCpG panels", PALETTE["rose"], PALETTE["rose_light"]),
        ("GEO external\nvalidation", PALETTE["purple"], PALETTE["purple_light"]),
    ]
    xs = np.linspace(0.04, 0.82, len(steps))
    for i, (text, color, fill) in enumerate(steps):
        box = FancyBboxPatch(
            (xs[i], 0.42),
            0.145,
            0.34,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.7,
            edgecolor=color,
            facecolor=fill,
        )
        ax_flow.add_patch(box)
        ax_flow.text(xs[i] + 0.0725, 0.59, text, ha="center", va="center", fontsize=7.4, color=PALETTE["ink"], fontweight="bold")
        if i < len(steps) - 1:
            arr = FancyArrowPatch(
                (xs[i] + 0.15, 0.59),
                (xs[i + 1] - 0.006, 0.59),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.7,
                color=PALETTE["muted"],
            )
            ax_flow.add_patch(arr)
    ax_flow.text(0.04, 0.88, "Leakage-free compact CpG-panel discovery workflow", fontsize=8.8, fontweight="bold", color=PALETTE["ink"])
    ax_flow.text(0.04, 0.19, "PathMethNet v2 is used as a complementary CpG-gene-pathway interpretation model.", fontsize=6.8, color=PALETTE["muted"])
    panel_label(ax_flow, "a")

    top_counts = counts.sort_values(ascending=False).head(12).sort_values()
    ax_tcga.barh(np.arange(len(top_counts)), top_counts.values, color=PALETTE["blue"], alpha=0.88)
    ax_tcga.set_yticks(np.arange(len(top_counts)))
    ax_tcga.set_yticklabels([x.replace("TCGA-", "") for x in top_counts.index])
    ax_tcga.set_xlabel("Samples")
    ax_tcga.set_title("Largest TCGA classes")
    clean_axis(ax_tcga, "x")
    ax_tcga.text(0.98, 0.08, f"{len(samples):,} tumors\n{counts.shape[0]} classes", transform=ax_tcga.transAxes, ha="right", va="bottom", fontsize=7, color=PALETTE["muted"])
    panel_label(ax_tcga, "b")

    names = [r[0].replace("_", " ") for r in cohort_rows]
    vals = [r[1] for r in cohort_rows]
    cols = [r[2] for r in cohort_rows]
    y = np.arange(len(names))
    ax_ext.barh(y, vals, color=cols, alpha=0.88)
    ax_ext.set_yticks(y)
    ax_ext.set_yticklabels(names)
    ax_ext.set_xlabel("Samples")
    ax_ext.set_title("Formal external cohorts")
    for yi, v in zip(y, vals):
        ax_ext.text(v + max(vals) * 0.02, yi, str(v), va="center", fontsize=6.8, color=PALETTE["ink"])
    ax_ext.set_xlim(0, max(vals) * 1.28)
    clean_axis(ax_ext, "x")
    panel_label(ax_ext, "c")
    fig.text(0.02, 0.985, "Study design and validation cohorts", ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])
    save(fig, "npj_fig1_study_design")


def fig2_internal_performance() -> None:
    panel = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/panel_performance_pretty.csv")
    baseline = pd.read_csv(RESULTS / "internal_cv/baseline_summary/all_baseline_model_comparison_pretty.csv")
    baseline = baseline[baseline["complete"].eq("yes")].head(8).copy()
    baseline["name"] = baseline["model"].astype(str) + "\n" + baseline["feature_method"].astype(str) + " " + baseline["top_k"].astype(str)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.35)
    ax_perf = fig.add_subplot(gs[0, :])
    ax_gain = fig.add_subplot(gs[1, 0])
    ax_base = fig.add_subplot(gs[1, 1])

    x = panel["panel_size"].to_numpy()
    ax_perf.plot(x, panel["accuracy_mean"], marker="o", color=PALETTE["blue"], lw=1.8, ms=4.2, label="Accuracy")
    ax_perf.fill_between(x, panel["accuracy_mean"] - panel["accuracy_std"], panel["accuracy_mean"] + panel["accuracy_std"], color=PALETTE["blue"], alpha=0.12, linewidth=0)
    ax_perf.plot(x, panel["macro_f1_mean"], marker="s", color=PALETTE["rose"], lw=1.8, ms=4.0, label="Macro F1")
    ax_perf.fill_between(x, panel["macro_f1_mean"] - panel["macro_f1_std"], panel["macro_f1_mean"] + panel["macro_f1_std"], color=PALETTE["rose"], alpha=0.12, linewidth=0)
    ax_perf.plot(x, panel["top3_accuracy_mean"], marker="^", color=PALETTE["teal"], lw=1.5, ms=4.2, label="Top-3")
    ax_perf.set_xscale("log")
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels([str(int(v)) for v in x])
    ax_perf.set_ylim(0.1, 1.02)
    ax_perf.set_xlabel("CpG panel size")
    ax_perf.set_ylabel("Performance")
    ax_perf.set_title("Compact panels preserve high cross-validated performance")
    ax_perf.legend(loc="lower right", ncol=3)
    clean_axis(ax_perf, "y")
    panel_label(ax_perf, "a")

    ref = float(panel.loc[panel["panel_size"].eq(1000), "macro_f1_mean"].iloc[0])
    retain = panel["macro_f1_mean"] / ref
    ax_gain.bar(np.arange(len(panel)), retain, color=PALETTE["orange"], alpha=0.82)
    ax_gain.axhline(0.95, color=PALETTE["ink"], lw=0.7, ls="--")
    ax_gain.set_xticks(np.arange(len(panel)))
    ax_gain.set_xticklabels(panel["panel_size"].astype(int), rotation=35, ha="right")
    ax_gain.set_ylim(0, 1.06)
    ax_gain.set_ylabel("Macro F1 retained\nvs 1000 CpGs")
    ax_gain.set_title("Diminishing returns after compact panels")
    clean_axis(ax_gain, "y")
    panel_label(ax_gain, "b")

    top = baseline.sort_values("macro_f1_mean", ascending=True)
    ax_base.errorbar(top["macro_f1_mean"], np.arange(len(top)), xerr=top["macro_f1_std"], fmt="o", color=PALETTE["blue"], ecolor=PALETTE["blue_light"], elinewidth=2.0, ms=4)
    ax_base.set_yticks(np.arange(len(top)))
    ax_base.set_yticklabels(top["name"], fontsize=5.8)
    ax_base.set_xlabel("Macro F1")
    ax_base.set_title("Best complete baseline runs")
    ax_base.set_xlim(max(0.75, top["macro_f1_mean"].min() - 0.04), 0.96)
    clean_axis(ax_base, "x")
    panel_label(ax_base, "c")
    fig.text(0.02, 0.985, "Internal TCGA cross-validation performance", ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])
    save(fig, "npj_fig2_internal_performance")


def fig3_external_validation() -> None:
    external = load_external()
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.14, 1], height_ratios=[1.08, 1], hspace=0.46, wspace=0.52)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_conf = fig.add_subplot(gs[1, :])

    pivot = external.pivot(index="label", columns="panel_size", values="top1")
    order = ["GSE56044\nLUAD/LUSC", "GSE48684\nCOAD/READ", "GSE53051\n5 lineages", "GSE105260\nKIRC"]
    pivot = pivot.loc[order]
    cmap = LinearSegmentedColormap.from_list("npj_acc", ["#F2F4F7", "#DCE9F6", "#89B9D9", "#2B6CB0", "#163D63"])
    im = ax_heat.imshow(pivot.values, vmin=0.6, vmax=1.0, cmap=cmap, aspect="auto")
    ax_heat.set_xticks(np.arange(len(pivot.columns)))
    ax_heat.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax_heat.set_yticks(np.arange(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index)
    ax_heat.set_xlabel("CpG panel size")
    ax_heat.set_title("Top-1 / exact accuracy across panel sizes")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color="white" if val > 0.86 else PALETTE["ink"])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cbar.ax.set_title("Acc.", fontsize=6.4, pad=3)
    panel_label(ax_heat, "a")

    anchor = external[external["panel_size"].eq(1000)].sort_values("top1")
    y = np.arange(len(anchor))
    ax_bar.barh(y, anchor["top1"], color=anchor["color"], alpha=0.88, height=0.55)
    ax_bar.scatter(anchor["top3"], y, marker="D", s=18, color=PALETTE["ink"], label="Top-3")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(anchor["label"])
    ax_bar.set_xlim(0.55, 1.03)
    ax_bar.set_xlabel("Accuracy")
    ax_bar.set_title("1000-CpG anchor panel")
    ax_bar.legend(loc="lower right")
    clean_axis(ax_bar, "x")
    panel_label(ax_bar, "b")

    for cohort, cfg in COHORTS.items():
        path = cfg["dir"] / "lr_panel/external_confidence_curve.tsv"
        curve = pd.read_csv(path, sep="\t")
        curve = curve[curve["panel_size"].eq(1000)]
        if curve.empty:
            continue
        ax_conf.plot(curve["coverage"], curve["accuracy_retained"], color=cfg["color"], lw=1.6, label=cfg["label"].replace("\n", " "))
    ax_conf.set_xlim(1.02, -0.02)
    ax_conf.set_ylim(0.55, 1.03)
    ax_conf.set_xlabel("Fraction of samples retained after confidence filtering")
    ax_conf.set_ylabel("Accuracy among retained samples")
    ax_conf.set_title("Confidence filtering supports decision-threshold reporting")
    ax_conf.legend(loc="lower left", ncol=2)
    clean_axis(ax_conf, "y")
    panel_label(ax_conf, "c")
    fig.text(0.02, 0.985, "Independent GEO validation", ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])
    save(fig, "npj_fig3_external_validation")


def fig4_biomarker_interpretation() -> None:
    stable = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/stable_cpg_panel_annotated.tsv", sep="\t")
    class_markers = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/class_specific_top20_cpg_pretty.tsv", sep="\t")
    pathmeth = pd.read_csv(RESULTS / "internal_cv/pathmethnet_v2_summary/pathmethnet_v2_summary_metrics.csv").rename(columns={"Unnamed: 0": "metric"})
    top100 = stable[stable["panel_size"].eq(100)].head(100).copy()

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.44, wspace=0.36)
    ax_prom = fig.add_subplot(gs[0, 0])
    ax_island = fig.add_subplot(gs[0, 1])
    ax_marker = fig.add_subplot(gs[1, 0])
    ax_pm = fig.add_subplot(gs[1, 1])

    promoter_count = int(top100["is_promoter"].fillna(False).sum())
    promoter = pd.Series({"Promoter": promoter_count, "Other": int(len(top100) - promoter_count)})
    ax_prom.pie(
        promoter.values,
        labels=promoter.index,
        colors=[PALETTE["teal"], PALETTE["gray"]],
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white", "width": 0.62},
        textprops={"fontsize": 7},
        autopct=lambda p: f"{p:.0f}%",
    )
    ax_prom.set_title("Top-100 CpG promoter context")
    panel_label(ax_prom, "a")

    island = top100["relation_to_cpg_island"].fillna("Unannotated").value_counts().head(6).sort_values()
    ax_island.barh(np.arange(len(island)), island.values, color=PALETTE["blue"], alpha=0.86)
    ax_island.set_yticks(np.arange(len(island)))
    ax_island.set_yticklabels(island.index)
    ax_island.set_xlabel("CpG count")
    ax_island.set_title("CpG island relation")
    clean_axis(ax_island, "x")
    panel_label(ax_island, "b")

    marker_summary = class_markers.groupby("label").agg(
        mean_abs_coef=("mean_abs_coef", "mean"),
        promoter_fraction=("is_promoter", "mean"),
    )
    marker_summary = marker_summary.sort_values("mean_abs_coef", ascending=False).head(12).sort_values("mean_abs_coef")
    ax_marker.scatter(marker_summary["mean_abs_coef"], np.arange(len(marker_summary)), s=20 + marker_summary["promoter_fraction"] * 90, color=PALETTE["rose"], alpha=0.78, edgecolor="white", linewidth=0.4)
    ax_marker.set_yticks(np.arange(len(marker_summary)))
    ax_marker.set_yticklabels([x.replace("TCGA-", "") for x in marker_summary.index])
    ax_marker.set_xlabel("Mean absolute coefficient")
    ax_marker.set_title("Cancer-specific marker strength")
    clean_axis(ax_marker, "x")
    ax_marker.text(0.98, 0.03, "point size: promoter fraction", transform=ax_marker.transAxes, ha="right", va="bottom", fontsize=6.2, color=PALETTE["muted"])
    panel_label(ax_marker, "c")

    keep = pathmeth[pathmeth["metric"].isin(["macro_f1", "balanced_accuracy", "accuracy", "top3_accuracy"])].copy()
    keep["metric"] = keep["metric"].replace({"macro_f1": "Macro F1", "balanced_accuracy": "BACC", "accuracy": "Accuracy", "top3_accuracy": "Top-3"})
    ax_pm.bar(np.arange(len(keep)), keep["mean"], yerr=keep["std"], color=[PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]], alpha=0.86, capsize=2.5)
    ax_pm.set_xticks(np.arange(len(keep)))
    ax_pm.set_xticklabels(keep["metric"], rotation=25, ha="right")
    ax_pm.set_ylim(0.75, 1.01)
    ax_pm.set_ylabel("Internal CV performance")
    ax_pm.set_title("PathMethNet v2 interpretation model")
    clean_axis(ax_pm, "y")
    panel_label(ax_pm, "d")
    fig.text(0.02, 0.985, "CpG biomarker annotation and hierarchical interpretation", ha="left", va="top", fontsize=9.2, fontweight="bold", color=PALETTE["ink"])
    save(fig, "npj_fig4_biomarker_interpretation")


def write_gallery() -> None:
    md = """# npj 风格论文图表总览

这一版图表按 `npj Precision Oncology` 常见排版重新绘制：复合 panel、小字号、低饱和配色、矢量 PDF、主图直接承载论文叙事。

## Main Figures

### Figure 1. Study design and validation cohorts

文件：[PNG](manuscript_figures_npj/npj_fig1_study_design.png) / [PDF](manuscript_figures_npj/npj_fig1_study_design.pdf)

![Figure 1](manuscript_figures_npj/npj_fig1_study_design.png)

### Figure 2. Internal TCGA cross-validation performance

文件：[PNG](manuscript_figures_npj/npj_fig2_internal_performance.png) / [PDF](manuscript_figures_npj/npj_fig2_internal_performance.pdf)

![Figure 2](manuscript_figures_npj/npj_fig2_internal_performance.png)

### Figure 3. Independent GEO validation

文件：[PNG](manuscript_figures_npj/npj_fig3_external_validation.png) / [PDF](manuscript_figures_npj/npj_fig3_external_validation.pdf)

![Figure 3](manuscript_figures_npj/npj_fig3_external_validation.png)

### Figure 4. CpG biomarker annotation and hierarchical interpretation

文件：[PNG](manuscript_figures_npj/npj_fig4_biomarker_interpretation.png) / [PDF](manuscript_figures_npj/npj_fig4_biomarker_interpretation.pdf)

![Figure 4](manuscript_figures_npj/npj_fig4_biomarker_interpretation.png)

## Main Tables

| Table | File | Role |
|---|---|---|
| Table 1. Cohort and data summary | [CSV](manuscript_tables/table_0_cohort_summary.csv) / [MD](manuscript_tables/table_0_cohort_summary.md) | Development and external validation cohorts |
| Table 2. Internal CV performance | [CSV](manuscript_tables/table_1_internal_cv_panel_performance.csv) / [MD](manuscript_tables/table_1_internal_cv_panel_performance.md) | Compact panel performance |
| Table 3. External validation at 1000 CpGs | [CSV](manuscript_tables/table_2_external_validation_1000_cpg.csv) / [MD](manuscript_tables/table_2_external_validation_1000_cpg.md) | Formal external validation |
| Table 4. Best external panel per cohort | [CSV](manuscript_tables/table_3_external_validation_best_panel.csv) / [MD](manuscript_tables/table_3_external_validation_best_panel.md) | Cohort-specific panel behavior |

## Supplementary Figures and Tables

保留上一版补图补表作为 supplementary material。正式主文优先使用本文件中的 4 张 `npj_` 复合图。

"""
    (RESULTS / "manuscript_figure_table_gallery_npj.md").write_text(md)


def main() -> None:
    set_style()
    fig1_study_design()
    fig2_internal_performance()
    fig3_external_validation()
    fig4_biomarker_interpretation()
    write_gallery()
    manifest = {
        "figures_dir": str(FIG_DIR),
        "figures": sorted(p.name for p in FIG_DIR.glob("npj_fig*.png")),
        "gallery": str(RESULTS / "manuscript_figure_table_gallery_npj.md"),
    }
    (RESULTS / "manuscript_figures_npj_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
