"""
Build per-class accuracy breakdown for GSE53051 multi-tissue external cohort.

Shows which tissue lineages are most/least reliably predicted at each panel size.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "results" / "manuscript_figures"
TBL_DIR = PROJECT_ROOT / "results" / "manuscript_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

NATURE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
LIGHT_GRAY = "#E6E6E6"

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

COHORT = "GSE53051"
PANEL_SIZES = [100, 200, 500, 1000]


def main():
    pred_path = (
        PROJECT_ROOT / "results" / "external_validation" / COHORT / "lr_panel" / "external_predictions.tsv"
    )
    if not pred_path.exists():
        print(f"Predictions not found: {pred_path}")
        return

    preds = pd.read_csv(pred_path, sep="\t")

    # Per-class accuracy by panel size
    rows = []
    for ps in PANEL_SIZES:
        sub = preds[preds["panel_size"] == ps]
        for label in sub["label"].unique():
            mask = sub["label"] == label
            n = mask.sum()
            n_correct = sub.loc[mask, "accepted_correct"].sum()
            rows.append({"panel_size": ps, "label": label, "n": n, "accuracy": n_correct / n if n > 0 else 0})

    per_class = pd.DataFrame(rows)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = sorted(per_class["label"].unique())
    x = np.arange(len(PANEL_SIZES))
    width = 0.15

    for i, label in enumerate(labels):
        vals = per_class[per_class["label"] == label].set_index("panel_size").loc[PANEL_SIZES, "accuracy"]
        ax.bar(x + i * width, vals, width, color=NATURE_COLORS[i % len(NATURE_COLORS)], label=label)

    ax.set_xlabel("Panel Size")
    ax.set_ylabel("Accepted Top-1 Accuracy")
    ax.set_title(f"{COHORT}: Per-Class Accuracy by Panel Size")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([str(s) for s in PANEL_SIZES])
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=LIGHT_GRAY, linewidth=0.8, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{COHORT}_per_class_accuracy.png", dpi=300)
    fig.savefig(FIG_DIR / f"{COHORT}_per_class_accuracy.pdf")
    plt.close(fig)
    print(f"Saved: {COHORT}_per_class_accuracy.png/pdf")

    # Detail table
    detail = per_class.pivot(index="label", columns="panel_size", values="accuracy").round(4)
    detail.columns = [str(c) for c in detail.columns]
    detail["n"] = per_class.groupby("label")["n"].first()
    detail = detail[["n"] + [str(s) for s in PANEL_SIZES]]
    detail.to_csv(TBL_DIR / f"{COHORT}_per_class_accuracy.tsv", sep="\t")
    print(f"Saved: {COHORT}_per_class_accuracy.tsv")
    print(detail.to_string())


if __name__ == "__main__":
    main()
