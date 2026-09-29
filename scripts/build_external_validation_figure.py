"""
Build external validation performance figure and summary table for manuscript.

Panel performance curves: accuracy vs panel size across GEO cohorts.
Accepts or exact accuracy depending on cohort mode.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "manuscript_figures"
TBL_DIR = RESULTS_DIR / "manuscript_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

NATURE = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
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

COHORT_DISPLAY = {
    "GSE56044": "GSE56044 (LUAD/LUSC)",
    "GSE48684": "GSE48684 (COAD/READ)",
    "GSE53051": "GSE53051 (Multi-tissue)",
    "GSE69914": "GSE69914 (BRCA provisional)",
    "GSE105260_official_series": "GSE105260 official (KIRC)",
}

COHORT_METRIC_KEY = {
    "GSE56044": "exact_accuracy",
    "GSE48684": "accepted_top1_accuracy",
    "GSE53051": "accepted_top1_accuracy",
    "GSE69914": "accepted_top1_accuracy",
    "GSE105260_official_series": "exact_accuracy",
}

COHORT_TOP3_KEY = {
    "GSE56044": "exact_top3_accuracy",
    "GSE48684": "accepted_top3_accuracy",
    "GSE53051": "accepted_top3_accuracy",
    "GSE69914": "accepted_top3_accuracy",
    "GSE105260_official_series": "exact_top3_accuracy",
}

COHORT_STYLE = {
    "GSE56044": {"marker": "o", "color": NATURE["blue"], "linestyle": "-"},
    "GSE48684": {"marker": "s", "color": NATURE["orange"], "linestyle": "--"},
    "GSE53051": {"marker": "^", "color": NATURE["green"], "linestyle": "-."},
    "GSE69914": {"marker": "D", "color": NATURE["gray"], "linestyle": ":"},
    "GSE105260_official_series": {"marker": "v", "color": NATURE["purple"], "linestyle": "--"},
}


def load_cohort_metrics(cohort: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / "external_validation" / cohort / "lr_panel" / "external_panel_metrics.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    df["cohort"] = cohort
    return df


def main():
    dfs = []
    for cohort in COHORT_DISPLAY:
        df = load_cohort_metrics(cohort)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("No external validation metrics found.")
        return

    all_metrics = pd.concat(dfs, ignore_index=True)
    panel_sizes = sorted(all_metrics["panel_size"].unique())

    # ---- Figure 1: top1 accuracy vs panel size ----
    fig, ax = plt.subplots(figsize=(7, 5))

    for cohort in COHORT_DISPLAY:
        sub = all_metrics[all_metrics["cohort"] == cohort]
        if sub.empty:
            continue
        key = COHORT_METRIC_KEY[cohort]
        values = sub[key].values
        style = COHORT_STYLE[cohort]
        ax.plot(
            panel_sizes,
            values,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=COHORT_DISPLAY[cohort],
            linewidth=1.8,
            markersize=7,
        )

    ax.set_xlabel("CpG Panel Size")
    ax.set_ylabel("Top-1 Accuracy / Exact Accuracy")
    ax.set_title("External Validation: Panel Size vs Performance")
    ax.set_xticks(panel_sizes)
    ax.set_xticklabels([str(s) for s in panel_sizes])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", color=NATURE["light_gray"], linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "external_validation_panel_curve.png", dpi=300)
    fig.savefig(FIG_DIR / "external_validation_panel_curve.pdf")
    plt.close(fig)
    print(f"Saved: external_validation_panel_curve.png/pdf")

    # ---- Figure 2: confidence coverage curves ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for cohort in COHORT_DISPLAY:
        curve_path = (
            RESULTS_DIR / "external_validation" / cohort / "lr_panel" / "external_confidence_curve.tsv"
        )
        if not curve_path.exists():
            continue
        curve = pd.read_csv(curve_path, sep="\t")
        curve_1000 = curve[curve["panel_size"] == 1000]
        if curve_1000.empty:
            continue
        style = COHORT_STYLE[cohort]
        ax.plot(
            curve_1000["threshold"],
            curve_1000["accuracy_retained"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            label=COHORT_DISPLAY[cohort],
            linewidth=1.5,
            markersize=4,
        )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy on Retained Samples")
    ax.set_title("1000-CpG Panel: Confidence vs Accuracy")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", color=NATURE["light_gray"], linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "external_validation_confidence_curve.png", dpi=300)
    fig.savefig(FIG_DIR / "external_validation_confidence_curve.pdf")
    plt.close(fig)
    print(f"Saved: external_validation_confidence_curve.png/pdf")

    # ---- Summary table ----
    summary_rows = []
    for cohort in COHORT_DISPLAY:
        sub = all_metrics[all_metrics["cohort"] == cohort]
        if sub.empty:
            continue
        row_1000 = sub[sub["panel_size"] == 1000]
        if row_1000.empty:
            continue
        r = row_1000.iloc[0]
        summary_rows.append(
            {
                "Cohort": COHORT_DISPLAY[cohort],
                "Mode": r.get("external_labels", ""),
                "N": int(r["n_external_samples"]),
                "Top1": f"{r[COHORT_METRIC_KEY[cohort]]:.4f}",
                "Top3": f"{r[COHORT_TOP3_KEY[cohort]]:.4f}",
                "MeanConf": f"{r['mean_confidence']:.4f}" if "mean_confidence" in r else "",
            }
        )
    tbl = pd.DataFrame(summary_rows)
    tbl.to_csv(TBL_DIR / "external_validation_summary.tsv", sep="\t", index=False)
    print(f"Saved: external_validation_summary.tsv")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
