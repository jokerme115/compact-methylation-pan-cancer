"""
Build confusion analysis for GSE56044: 1000 CpG panel confusion matrix and error breakdown.

GSE56044 is the strongest external validation cohort with exact LUAD/LUSC labels.
This script identifies the most common misclassification patterns.
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COHORT = "GSE56044"
PANEL_SIZE = 1000


def main():
    pred_path = (
        PROJECT_ROOT
        / "results"
        / "external_validation"
        / COHORT
        / "lr_panel"
        / "external_predictions.tsv"
    )
    if not pred_path.exists():
        print(f"Predictions not found: {pred_path}")
        return

    preds = pd.read_csv(pred_path, sep="\t")
    preds = preds[preds["panel_size"] == PANEL_SIZE].copy()
    if preds.empty:
        print(f"No predictions for panel_size={PANEL_SIZE}")
        return

    # ---- Confusion matrix ----
    true_labels = preds["label"].astype(str).to_numpy()
    predicted_labels = preds["pred_label"].astype(str).to_numpy()
    classes = sorted(set(true_labels) | set(predicted_labels))

    cm = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for t, p in zip(true_labels, predicted_labels):
        cm.loc[t, p] += 1

    # Normalize by row (true class)
    cm_norm = cm.div(cm.sum(axis=1), axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm.values, cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{COHORT}: 1000-CpG Panel Confusion Matrix\n(n={len(preds)})")

    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm.values[i, j]
            color = "white" if val < 0.78 else "#222222"
            ax.text(j, i, f"{val:.2f}\n({cm.values[i, j]})", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.82, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{COHORT}_confusion_panel{PANEL_SIZE}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{COHORT}_confusion_panel{PANEL_SIZE}.pdf")
    plt.close(fig)
    print(f"Saved: {COHORT}_confusion_panel{PANEL_SIZE}.png/pdf")

    # ---- Error breakdown table ----
    errors = preds[preds["exact_correct"] == False].copy()
    print(f"\nErrors (n={len(errors)}):")
    for _, row in errors.iterrows():
        top3 = "; ".join(
            [
                f"{row['top1_label']}({row['top1_prob']:.3f})",
                f"{row['top2_label']}({row['top2_prob']:.3f})",
                f"{row['top3_label']}({row['top3_prob']:.3f})",
            ]
        )
        print(f"  {row['sample_id']}: true={row['label']}, pred={row['pred_label']}, conf={row['confidence']:.3f}")
        print(f"    Top3: {top3}")

    error_rows = []
    for _, row in errors.iterrows():
        error_rows.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["label"],
                "histology": row["histology"],
                "pred_label": row["pred_label"],
                "confidence": round(row["confidence"], 4),
                "top1": f"{row['top1_label']}({row['top1_prob']:.3f})",
                "top2": f"{row['top2_label']}({row['top2_prob']:.3f})",
                "top3": f"{row['top3_label']}({row['top3_prob']:.3f})",
            }
        )
    if error_rows:
        err_df = pd.DataFrame(error_rows)
        err_df.to_csv(TBL_DIR / f"{COHORT}_error_breakdown.tsv", sep="\t", index=False)
        print(f"\nSaved: {COHORT}_error_breakdown.tsv")

    # ---- Top mismatches ----
    print(f"\nCorrect: {preds['exact_correct'].sum()}/{len(preds)}")
    pairs = preds.groupby(["label", "pred_label"]).size().reset_index(name="count")
    pairs = pairs[pairs["label"] != pairs["pred_label"]].sort_values("count", ascending=False)
    print("Misclassification pairs:")
    for _, row in pairs.iterrows():
        print(f"  {row['label']} -> {row['pred_label']}: {row['count']}")


if __name__ == "__main__":
    main()
