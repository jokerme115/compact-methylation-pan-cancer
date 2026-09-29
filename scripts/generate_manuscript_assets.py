from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLE_DIR = RESULTS / "manuscript_tables"
FIG_DIR = RESULTS / "manuscript_figures"

NATURE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
}
LINE_COLORS = [NATURE["blue"], NATURE["vermillion"], NATURE["green"], NATURE["purple"], NATURE["orange"], NATURE["sky"]]


FORMAL_EXTERNAL = {
    "GSE56044": {
        "label": "GSE56044\nLUAD/LUSC",
        "mode": "Exact LUAD/LUSC",
        "dir": RESULTS / "external_validation/GSE56044",
        "metric": "exact_accuracy",
        "top3": "exact_top3_accuracy",
        "paper_note": "independent lung cancer cohort",
    },
    "GSE48684": {
        "label": "GSE48684\nCOAD/READ",
        "mode": "COAD/READ lineage",
        "dir": RESULTS / "external_validation/GSE48684",
        "metric": "accepted_top1_accuracy",
        "top3": "accepted_top3_accuracy",
        "paper_note": "independent colorectal cancer cohort",
    },
    "GSE53051": {
        "label": "GSE53051\n5 lineages",
        "mode": "Multi-tissue lineage",
        "dir": RESULTS / "external_validation/GSE53051",
        "metric": "accepted_top1_accuracy",
        "top3": "accepted_top3_accuracy",
        "paper_note": "multi-tissue external cohort",
    },
    "GSE105260 official": {
        "label": "GSE105260\nKIRC",
        "mode": "Exact KIRC from official beta",
        "dir": RESULTS / "external_validation/GSE105260_official_series",
        "metric": "exact_accuracy",
        "top3": "exact_top3_accuracy",
        "paper_note": "official minfi/SWAN processed ccRCC cohort",
    },
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": NATURE["black"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.md").write_text(to_markdown(df))


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def fmt(value: object, col: str) -> str:
        if pd.isna(value):
            return ""
        if any(token in col.lower() for token in ["size", "samples", "count"]):
            try:
                return str(int(value))
            except (TypeError, ValueError):
                return str(value)
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col], col) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_external_metrics() -> pd.DataFrame:
    rows = []
    for cohort, cfg in FORMAL_EXTERNAL.items():
        metrics = pd.read_csv(cfg["dir"] / "lr_panel/external_panel_metrics.tsv", sep="\t")
        manifest = json.loads((cfg["dir"] / "dataset_manifest.json").read_text())
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "cohort": cohort,
                    "cohort_label": cfg["label"],
                    "evaluation_mode": cfg["mode"],
                    "panel_size": int(row["panel_size"]),
                    "n_samples": int(row["n_external_samples"]),
                    "primary_accuracy": float(row[cfg["metric"]]),
                    "top3_accuracy": float(row[cfg["top3"]]),
                    "mean_confidence": float(row["mean_confidence"]),
                    "labels": row["external_labels"],
                    "manifest_label_counts": json.dumps(manifest.get("label_counts", {}), ensure_ascii=False),
                    "note": cfg["paper_note"],
                }
            )
    return pd.DataFrame(rows)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    tcga_samples = pd.read_csv(ROOT / "data/tcga_450k/samples_locked.tsv", sep="\t")
    tcga_classes = sorted(tcga_samples["project_id"].dropna().unique())
    cohort_rows = [
        {
            "Cohort": "TCGA candidate manifest",
            "Role": "Source audit",
            "Samples": 9812,
            "Classes / labels": "33 TCGA classes; 747 normal rows",
            "Evaluation": "Normal rows excluded before formal training",
            "Use in manuscript": "Provenance audit only",
        },
        {
            "Cohort": "TCGA 450k",
            "Role": "Model development and internal validation",
            "Samples": int(len(tcga_samples)),
            "Classes / labels": f"{len(tcga_classes)} TCGA cancer classes",
            "Evaluation": "Patient-level stratified 5-fold x 3 repeated CV",
            "Use in manuscript": "Primary training and internal validation cohort",
        }
    ]
    for cohort, cfg in FORMAL_EXTERNAL.items():
        manifest = json.loads((cfg["dir"] / "dataset_manifest.json").read_text())
        labels = manifest.get("label_counts", {})
        cohort_rows.append(
            {
                "Cohort": cohort,
                "Role": "Formal external validation",
                "Samples": int(manifest.get("n_kept_samples", sum(labels.values()))),
                "Classes / labels": "; ".join(f"{k}: {v}" for k, v in labels.items()),
                "Evaluation": cfg["mode"],
                "Use in manuscript": cfg["paper_note"],
            }
        )
    cohort_table = pd.DataFrame(cohort_rows)
    save_table(cohort_table, "table_0_cohort_summary")

    panel = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/panel_performance_pretty.csv")
    panel_table = panel[
        [
            "panel_size",
            "accuracy_mean",
            "accuracy_std",
            "balanced_accuracy_mean",
            "macro_f1_mean",
            "macro_f1_std",
            "top3_accuracy_mean",
            "top5_accuracy_mean",
            "ece_15_bins_mean",
        ]
    ].copy()
    panel_table.columns = [
        "CpG panel size",
        "Accuracy mean",
        "Accuracy SD",
        "Balanced accuracy mean",
        "Macro F1 mean",
        "Macro F1 SD",
        "Top-3 accuracy mean",
        "Top-5 accuracy mean",
        "ECE mean",
    ]
    save_table(panel_table, "table_1_internal_cv_panel_performance")

    external = load_external_metrics()
    external_1000 = external[external["panel_size"].eq(1000)].copy()
    max_primary = external.groupby("cohort")["primary_accuracy"].transform("max")
    external_best = external[external["primary_accuracy"].eq(max_primary)].copy()
    external_summary = external_1000[
        [
            "cohort",
            "evaluation_mode",
            "n_samples",
            "labels",
            "primary_accuracy",
            "top3_accuracy",
            "mean_confidence",
            "note",
        ]
    ].copy()
    external_summary = external_summary.rename(
        columns={
            "cohort": "Cohort",
            "evaluation_mode": "Evaluation",
            "n_samples": "Samples",
            "labels": "Labels",
            "primary_accuracy": "Top-1/exact accuracy",
            "top3_accuracy": "Top-3 accuracy",
            "mean_confidence": "Mean confidence",
            "note": "Use",
        }
    )
    save_table(external_summary, "table_2_external_validation_1000_cpg")

    best_table = external_best[
        [
            "cohort",
            "evaluation_mode",
            "panel_size",
            "n_samples",
            "primary_accuracy",
            "top3_accuracy",
            "mean_confidence",
        ]
    ].rename(
        columns={
            "cohort": "Cohort",
            "evaluation_mode": "Evaluation",
            "panel_size": "Best panel size",
            "n_samples": "Samples",
            "primary_accuracy": "Best top-1/exact accuracy",
            "top3_accuracy": "Top-3 accuracy at best panel",
            "mean_confidence": "Mean confidence",
        }
    )
    save_table(best_table, "table_3_external_validation_best_panel")

    stable = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/stable_cpg_panel_annotated.tsv", sep="\t")
    top100 = stable[stable["panel_size"].eq(100)].head(100).copy()
    top100 = top100[
        [
            "probe_id",
            "mean_rank",
            "selected_count",
            "gene_symbol",
            "gene_region",
            "relation_to_cpg_island",
            "is_promoter",
            "top_pathways",
        ]
    ]
    save_table(top100, "supp_table_stable_top100_cpg_panel")

    class_markers = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/class_specific_top20_cpg_pretty.tsv", sep="\t")
    save_table(class_markers, "supp_table_class_specific_top20_cpg_markers")

    pathmeth = pd.read_csv(RESULTS / "internal_cv/pathmethnet_v2_summary/pathmethnet_v2_summary_metrics.csv")
    pathmeth = pathmeth.rename(columns={"Unnamed: 0": "metric"})
    save_table(pathmeth, "supp_table_pathmethnet_v2_metrics")

    gse69914_path = RESULTS / "external_validation/GSE69914/lr_panel/external_panel_metrics.tsv"
    if gse69914_path.exists():
        gse69914 = pd.read_csv(gse69914_path, sep="\t")
        gse69914.insert(0, "cohort", "GSE69914 provisional BRCA")
        save_table(gse69914, "supp_table_gse69914_provisional_brca")

    custom_path = RESULTS / "external_validation/GSE105260/gse105260_diagnostic_by_panel.tsv"
    if custom_path.exists():
        custom = pd.read_csv(custom_path, sep="\t")
        save_table(custom, "supp_table_gse105260_custom_idat_diagnostic")

    return panel, external


def plot_workflow_schematic() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.58, 0.16, 0.24, "TCGA 450k\n9065 tumors\n33 cancer classes", NATURE["blue"]),
        (0.24, 0.58, 0.16, 0.24, "Patient-level CV\n5 folds x 3 repeats\nno sample leakage", NATURE["sky"]),
        (0.46, 0.58, 0.16, 0.24, "Fold-wise feature\nselection\ncompact CpG panels", NATURE["green"]),
        (0.68, 0.58, 0.16, 0.24, "LR panel model\n100-1000 CpGs\ncalibrated scores", NATURE["orange"]),
        (0.82, 0.18, 0.16, 0.24, "GEO validation\nlung, CRC, mixed,\nkidney cohorts", NATURE["vermillion"]),
        (0.46, 0.18, 0.16, 0.24, "PathMethNet v2\nCpG-gene-pathway\ninterpretation", NATURE["purple"]),
        (0.68, 0.18, 0.16, 0.24, "Manuscript assets\nfigures, tables,\nbiomarker panels", NATURE["gray"]),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=9, weight="bold")
    arrows = [
        ((0.18, 0.70), (0.24, 0.70)),
        ((0.40, 0.70), (0.46, 0.70)),
        ((0.62, 0.70), (0.68, 0.70)),
        ((0.76, 0.58), (0.84, 0.42)),
        ((0.54, 0.58), (0.54, 0.42)),
        ((0.62, 0.30), (0.68, 0.30)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": NATURE["black"]})
    ax.text(0.02, 0.94, "Leakage-free compact CpG panel discovery and external validation workflow", fontsize=12, weight="bold", color=NATURE["black"])
    ax.text(0.02, 0.08, "Formal external validation uses GSE56044, GSE48684, GSE53051, and GSE105260 official series matrix.", fontsize=8, color=NATURE["gray"])
    save_fig(fig, "fig0_study_design_workflow")


def plot_internal_panel(panel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = panel["panel_size"]
    ax.errorbar(
        x,
        panel["macro_f1_mean"],
        yerr=panel["macro_f1_std"],
        marker="o",
        color=NATURE["blue"],
        linewidth=2.0,
        capsize=3,
        label="Macro F1",
    )
    ax.errorbar(
        x,
        panel["accuracy_mean"],
        yerr=panel["accuracy_std"],
        marker="s",
        color=NATURE["vermillion"],
        linewidth=2.0,
        capsize=3,
        label="Accuracy",
    )
    ax.plot(x, panel["top3_accuracy_mean"], marker="^", color=NATURE["green"], linewidth=2.0, label="Top-3 accuracy")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Internal cross-validation performance")
    ax.set_title("Compact CpG panels retain pan-cancer classification performance")
    ax.grid(True, axis="y", color=NATURE["light_gray"], linewidth=0.8)
    ax.legend(loc="lower right")
    save_fig(fig, "fig1_internal_cv_panel_performance")


def plot_external_heatmaps(external: pd.DataFrame) -> None:
    pivot1 = external.pivot(index="cohort_label", columns="panel_size", values="primary_accuracy")
    pivot3 = external.pivot(index="cohort_label", columns="panel_size", values="top3_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, pivot, title in [
        (axes[0], pivot1, "Top-1 / exact accuracy"),
        (axes[1], pivot3, "Top-3 accuracy"),
    ]:
        im = ax.imshow(pivot.values, vmin=0.55, vmax=1.0, cmap="cividis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("CpG panel size")
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.78 else NATURE["black"], fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Accuracy")
    save_fig(fig, "fig2_external_validation_heatmap")


def plot_external_1000_bar(external: pd.DataFrame) -> None:
    df = external[external["panel_size"].eq(1000)].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    order = df.sort_values("primary_accuracy", ascending=True)
    y = np.arange(len(order))
    ax.barh(y - 0.18, order["primary_accuracy"], height=0.34, color=NATURE["blue"], label="Top-1 / exact")
    ax.barh(y + 0.18, order["top3_accuracy"], height=0.34, color=NATURE["orange"], label="Top-3")
    ax.set_yticks(y)
    ax.set_yticklabels(order["cohort_label"])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("External validation performance")
    ax.set_title("Independent GEO validation of the 1000-CpG panel")
    ax.grid(True, axis="x", color=NATURE["light_gray"], linewidth=0.8)
    ax.legend(loc="lower right")
    save_fig(fig, "fig3_external_validation_1000_panel")


def plot_gse105260_panel() -> None:
    metrics = pd.read_csv(RESULTS / "external_validation/GSE105260_official_series/lr_panel/external_panel_metrics.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(metrics["panel_size"], metrics["exact_accuracy"], marker="o", color=NATURE["blue"], linewidth=2.0, label="Exact KIRC accuracy")
    ax.plot(metrics["panel_size"], metrics["exact_top3_accuracy"], marker="s", color=NATURE["orange"], linewidth=2.0, label="Top-3 accuracy")
    ax.set_xscale("log")
    ax.set_xticks(metrics["panel_size"])
    ax.set_xticklabels([str(int(v)) for v in metrics["panel_size"]])
    ax.set_ylim(0.55, 1.0)
    ax.set_xlabel("CpG panel size")
    ax.set_ylabel("Accuracy")
    ax.set_title("GSE105260 official beta validation favors the 100-CpG panel")
    ax.grid(True, axis="y", color=NATURE["light_gray"], linewidth=0.8)
    ax.legend()
    save_fig(fig, "supp_fig_gse105260_official_series_by_panel")


def plot_annotation_summary() -> None:
    stable = pd.read_csv(RESULTS / "internal_cv/lr_cpg_panel/stable_cpg_panel_annotated.tsv", sep="\t")
    top100 = stable[stable["panel_size"].eq(100)].head(100).copy()
    promoter_counts = top100["is_promoter"].fillna(False).map({True: "Promoter", False: "Non-promoter"}).value_counts()
    island_counts = top100["relation_to_cpg_island"].fillna("Unannotated").value_counts().head(8)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].bar(promoter_counts.index, promoter_counts.values, color=[NATURE["blue"], NATURE["gray"]])
    axes[0].set_title("Top-100 CpG promoter annotation")
    axes[0].set_ylabel("CpG count")
    axes[0].grid(True, axis="y", color=NATURE["light_gray"], linewidth=0.8)
    axes[1].barh(island_counts.index[::-1], island_counts.values[::-1], color=NATURE["green"])
    axes[1].set_title("Top-100 CpG island relation")
    axes[1].set_xlabel("CpG count")
    axes[1].grid(True, axis="x", color=NATURE["light_gray"], linewidth=0.8)
    save_fig(fig, "fig4_top100_cpg_annotation_summary")


def plot_pathmethnet_summary() -> None:
    pathmeth = pd.read_csv(RESULTS / "internal_cv/pathmethnet_v2_summary/pathmethnet_v2_summary_metrics.csv")
    pathmeth = pathmeth.rename(columns={"Unnamed: 0": "metric"})
    keep = pathmeth[pathmeth["metric"].isin(["macro_f1", "balanced_accuracy", "accuracy", "top3_accuracy"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(keep["metric"], keep["mean"], yerr=keep["std"], capsize=4, color=LINE_COLORS[: len(keep)])
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("Internal cross-validation performance")
    ax.set_title("PathMethNet v2 provides complementary interpretable performance")
    ax.grid(True, axis="y", color=NATURE["light_gray"], linewidth=0.8)
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "supp_fig_pathmethnet_v2_summary")


def write_readme() -> None:
    readme = """# Manuscript Tables and Figures

Generated by `project_a_paper/scripts/generate_manuscript_assets.py`.

## Main Tables

- `table_1_internal_cv_panel_performance.csv`: TCGA internal CV performance by CpG panel size.
- `table_2_external_validation_1000_cpg.csv`: Formal external validation at the 1000-CpG panel.
- `table_3_external_validation_best_panel.csv`: Best observed panel size per external cohort.

## Main Figures

- `fig1_internal_cv_panel_performance.png/pdf`: Internal CV panel-size trade-off.
- `fig2_external_validation_heatmap.png/pdf`: External validation across panel sizes.
- `fig3_external_validation_1000_panel.png/pdf`: 1000-CpG external validation bar plot.
- `fig4_top100_cpg_annotation_summary.png/pdf`: Top-100 CpG annotation summary.

## Supplementary Assets

- GSE105260 official series panel curve.
- PathMethNet v2 summary figure.
- Stable CpG and class-specific marker supplementary tables.
- GSE69914 provisional BRCA and custom IDAT diagnostic tables.
"""
    (RESULTS / "manuscript_assets_README.md").write_text(readme)


def table_link(stem: str) -> str:
    return f"[CSV](manuscript_tables/{stem}.csv) / [MD](manuscript_tables/{stem}.md)"


def write_gallery_md() -> None:
    gallery = f"""# 论文图表归类总览

本文件把已经生成的主文图、主文表、补充图和补充表集中归类。图片使用统一 Nature/npj 风格配色：正式结果使用 blue/orange/green/vermillion/purple，探索性或非正式结果使用 gray，热图使用 `cividis`。

## 主文图

### Fig. 1. Study design and model workflow

文件：[PNG](manuscript_figures/fig0_study_design_workflow.png) / [PDF](manuscript_figures/fig0_study_design_workflow.pdf)

用途：作为论文第一张流程图，说明 TCGA 450k、patient-level CV、fold-wise feature selection、LR compact CpG panel、GEO external validation 和 PathMethNet interpretation 的整体关系。

![Fig. 1 workflow](manuscript_figures/fig0_study_design_workflow.png)

### Fig. 2. Internal TCGA panel-size performance

文件：[PNG](manuscript_figures/fig1_internal_cv_panel_performance.png) / [PDF](manuscript_figures/fig1_internal_cv_panel_performance.pdf)

用途：展示 compact CpG panel 的性能曲线。主文应强调 100/200/500/1000 CpG 的 trade-off，不夸大 10 CpG panel。

![Fig. 2 internal performance](manuscript_figures/fig1_internal_cv_panel_performance.png)

### Fig. 3. External validation heatmap

文件：[PNG](manuscript_figures/fig2_external_validation_heatmap.png) / [PDF](manuscript_figures/fig2_external_validation_heatmap.pdf)

用途：展示正式 GEO 外部验证队列在不同 panel size 下的 top-1/exact 和 top-3 accuracy。

![Fig. 3 external heatmap](manuscript_figures/fig2_external_validation_heatmap.png)

### Fig. 4. External validation at the 1000-CpG anchor panel

文件：[PNG](manuscript_figures/fig3_external_validation_1000_panel.png) / [PDF](manuscript_figures/fig3_external_validation_1000_panel.pdf)

用途：突出 anchor panel 在 lung、colorectal、mixed solid tumor 和 kidney external cohorts 上的迁移表现。

![Fig. 4 external anchor](manuscript_figures/fig3_external_validation_1000_panel.png)

### Fig. 5. Top-100 CpG annotation summary

文件：[PNG](manuscript_figures/fig4_top100_cpg_annotation_summary.png) / [PDF](manuscript_figures/fig4_top100_cpg_annotation_summary.pdf)

用途：支持 biomarker 叙事，说明 compact panel 中 CpG 的 promoter/CpG-island annotation。

![Fig. 5 annotation](manuscript_figures/fig4_top100_cpg_annotation_summary.png)

## 主文表

| 表号 | 表名 | 文件 | 用途 |
|---|---|---|---|
| Table 1 | Cohort and data summary | {table_link("table_0_cohort_summary")} | 交代 TCGA 和正式 GEO 队列的样本量、标签和验证模式。 |
| Table 2 | Internal CV performance by CpG panel size | {table_link("table_1_internal_cv_panel_performance")} | 主性能表，列出 accuracy、balanced accuracy、macro F1、Top-3/Top-5、ECE。 |
| Table 3 | Formal external validation at 1000 CpGs | {table_link("table_2_external_validation_1000_cpg")} | 正式外部队列在 1000-CpG anchor panel 下的表现。 |
| Table 4 | Best panel size per external cohort | {table_link("table_3_external_validation_best_panel")} | 说明每个外部队列最佳 panel size，尤其是 GSE105260 official 100 CpG 最佳。 |

## 补充图

### Supplementary Fig. 1. GSE105260 official series panel curve

文件：[PNG](manuscript_figures/supp_fig_gse105260_official_series_by_panel.png) / [PDF](manuscript_figures/supp_fig_gse105260_official_series_by_panel.pdf)

![Supp Fig. 1 GSE105260](manuscript_figures/supp_fig_gse105260_official_series_by_panel.png)

### Supplementary Fig. 2. PathMethNet v2 summary metrics

文件：[PNG](manuscript_figures/supp_fig_pathmethnet_v2_summary.png) / [PDF](manuscript_figures/supp_fig_pathmethnet_v2_summary.pdf)

![Supp Fig. 2 PathMethNet](manuscript_figures/supp_fig_pathmethnet_v2_summary.png)

### Supplementary Fig. 3. GSE56044 confusion matrix

文件：[PNG](manuscript_figures/GSE56044_confusion_panel1000.png) / [PDF](manuscript_figures/GSE56044_confusion_panel1000.pdf)

![Supp Fig. 3 GSE56044 confusion](manuscript_figures/GSE56044_confusion_panel1000.png)

### Supplementary Fig. 4. GSE53051 per-class accuracy

文件：[PNG](manuscript_figures/GSE53051_per_class_accuracy.png) / [PDF](manuscript_figures/GSE53051_per_class_accuracy.pdf)

![Supp Fig. 4 GSE53051 per-class](manuscript_figures/GSE53051_per_class_accuracy.png)

### Supplementary Fig. 5. External confidence curve

文件：[PNG](manuscript_figures/external_validation_confidence_curve.png) / [PDF](manuscript_figures/external_validation_confidence_curve.pdf)

![Supp Fig. 5 confidence](manuscript_figures/external_validation_confidence_curve.png)

### Supplementary Fig. 6. External panel-size curve

文件：[PNG](manuscript_figures/external_validation_panel_curve.png) / [PDF](manuscript_figures/external_validation_panel_curve.pdf)

![Supp Fig. 6 panel curve](manuscript_figures/external_validation_panel_curve.png)

## 补充表

| 表号 | 表名 | 文件 | 用途 |
|---|---|---|---|
| Supplementary Table 1 | Stable top-100 CpG panel | {table_link("supp_table_stable_top100_cpg_panel")} | compact biomarker panel 核心列表。 |
| Supplementary Table 2 | Class-specific top-20 CpG markers | {table_link("supp_table_class_specific_top20_cpg_markers")} | 每个癌种的 top CpG marker。 |
| Supplementary Table 3 | PathMethNet v2 metrics | {table_link("supp_table_pathmethnet_v2_metrics")} | 解释模型性能补充。 |
| Supplementary Table 4 | GSE69914 provisional BRCA prediction | {table_link("supp_table_gse69914_provisional_brca")} | phenotype 不完整队列，只作为 sanity check。 |
| Supplementary Table 5 | GSE105260 custom IDAT diagnostic | {table_link("supp_table_gse105260_custom_idat_diagnostic")} | 说明 custom IDAT route 不作为正式结果。 |
| Supplementary Table 6 | GSE56044 error breakdown | [TSV](manuscript_tables/GSE56044_error_breakdown.tsv) | LUAD/LUSC 误分类个案分析。 |
| Supplementary Table 7 | GSE53051 per-class accuracy | [TSV](manuscript_tables/GSE53051_per_class_accuracy.tsv) | multi-tissue external cohort 的 class-level accuracy。 |

## 仍需补强

- PathMethNet 目前只有 summary metrics 图；主文若强调 biological interpretation，还需要从 attention/pathway 结果生成 CpG-gene-pathway 图。
- top CpG heatmap by cancer class 可作为后续增强图，用于连接 biomarker panel 和 cancer-specific methylation signatures。

"""
    (RESULTS / "manuscript_figure_table_gallery.md").write_text(gallery)


def main() -> None:
    ensure_dirs()
    panel, external = build_tables()
    plot_workflow_schematic()
    plot_internal_panel(panel)
    plot_external_heatmaps(external)
    plot_external_1000_bar(external)
    plot_gse105260_panel()
    plot_annotation_summary()
    plot_pathmethnet_summary()
    write_readme()
    write_gallery_md()
    manifest = {
        "tables_dir": str(TABLE_DIR),
        "figures_dir": str(FIG_DIR),
        "n_tables": len(list(TABLE_DIR.glob("*"))),
        "n_figures": len(list(FIG_DIR.glob("*"))),
    }
    (RESULTS / "manuscript_assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
