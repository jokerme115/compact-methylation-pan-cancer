from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed Q1 baseline predictions for one selected setting.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--feature-method", default="variance")
    parser.add_argument("--top-k", type=int, default=1000)
    return parser.parse_args()


def run_prefix(model: str, method: str, top_k: int) -> str:
    return f"{model}__{method}__top{top_k}__"


def load_labels(splits_dir: Path) -> list[str]:
    summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    return list(summary["labels"])


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    predictions_dir = baseline_dir / "predictions"
    features_dir = baseline_dir / "selected_features"
    splits_dir = Path(args.splits_dir)
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(splits_dir)
    label_to_id = {label: i for i, label in enumerate(labels)}
    prefix = run_prefix(args.model, args.feature_method, args.top_k)

    pred_files = sorted(predictions_dir.glob(f"{prefix}*.tsv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction TSV files match {prefix} in {predictions_dir}")

    pred_frames: list[pd.DataFrame] = []
    for path in pred_files:
        df = pd.read_csv(path, sep="\t")
        df["source_file"] = path.name
        pred_frames.append(df)
    pred = pd.concat(pred_frames, ignore_index=True)
    pred["true_id"] = pred["true_label"].map(label_to_id)
    pred["pred_id"] = pred["pred_label"].map(label_to_id)
    pred = pred.dropna(subset=["true_id", "pred_id"]).copy()
    pred["true_id"] = pred["true_id"].astype(int)
    pred["pred_id"] = pred["pred_id"].astype(int)
    pred.to_csv(out_dir / "best_model_predictions_long.tsv", sep="\t", index=False)

    y_true = pred["true_id"].to_numpy()
    y_pred = pred["pred_id"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "best_model_confusion_matrix_counts.csv")
    cm_norm = cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    cm_norm.to_csv(out_dir / "best_model_confusion_matrix_row_normalized.csv")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "support_repeated_cv": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).sort_values("f1", ascending=True)
    per_class.to_csv(out_dir / "best_model_per_class_metrics.csv", index=False)

    pairs = []
    for i, true_label in enumerate(labels):
        row_total = cm[i].sum()
        if row_total == 0:
            continue
        for j, pred_label in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append(
                {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "count": int(cm[i, j]),
                    "rate_within_true": float(cm[i, j] / row_total),
                }
            )
    pairs_df = pd.DataFrame(pairs).sort_values(["count", "rate_within_true"], ascending=False)
    pairs_df.to_csv(out_dir / "confusion_pairs_top.csv", index=False)

    focus = {"LUAD", "LUSC", "KIRC", "KIRP", "KICH", "COAD", "READ", "GBM", "LGG", "UCEC", "UCS"}
    focus_pairs = pairs_df[
        pairs_df["true_label"].str.replace("TCGA-", "", regex=False).isin(focus)
        | pairs_df["pred_label"].str.replace("TCGA-", "", regex=False).isin(focus)
    ]
    focus_pairs.to_csv(out_dir / "confusion_pairs_focus.csv", index=False)

    probe_ids = pd.read_csv(matrix_dir / "probe_ids.tsv", sep="\t")["probe_id"].astype(str).to_numpy()
    feature_files = sorted(features_dir.glob(f"{prefix}*.npy"))
    counts: Counter[int] = Counter()
    for path in feature_files:
        arr = np.load(path)
        counts.update(int(i) for i in arr)
    feature_rows = []
    n_folds = len(feature_files)
    for idx, count in counts.most_common():
        feature_rows.append(
            {
                "probe_index": idx,
                "probe_id": probe_ids[idx] if 0 <= idx < len(probe_ids) else "",
                "selected_count": count,
                "selected_fraction": count / max(n_folds, 1),
            }
        )
    pd.DataFrame(feature_rows).to_csv(out_dir / "selected_cpg_frequency.tsv", sep="\t", index=False)
    pd.DataFrame(feature_rows[:1000]).to_csv(out_dir / "selected_cpg_top1000.tsv", sep="\t", index=False)

    notes = [
        "# Baseline Analysis",
        "",
        f"- Setting: `{args.model} + {args.feature_method} + top{args.top_k}`",
        f"- Prediction folds loaded: `{len(pred_files)}`",
        f"- Repeated-CV prediction rows: `{len(pred)}`",
        f"- Feature files loaded: `{n_folds}`",
        "",
        "## Lowest Per-Class F1",
        "",
        per_class.head(12).to_csv(index=False),
        "",
        "## Top Confusion Pairs",
        "",
        pairs_df.head(20).to_csv(index=False),
    ]
    (out_dir / "baseline_result_notes.md").write_text("\n".join(notes), encoding="utf-8")

    manifest = {
        "baseline_dir": str(baseline_dir.resolve()),
        "matrix_dir": str(matrix_dir.resolve()),
        "splits_dir": str(splits_dir.resolve()),
        "setting": {"model": args.model, "feature_method": args.feature_method, "top_k": args.top_k},
        "prediction_files": len(pred_files),
        "feature_files": n_folds,
        "outputs": [
            "best_model_predictions_long.tsv",
            "best_model_confusion_matrix_counts.csv",
            "best_model_confusion_matrix_row_normalized.csv",
            "best_model_per_class_metrics.csv",
            "confusion_pairs_top.csv",
            "confusion_pairs_focus.csv",
            "selected_cpg_frequency.tsv",
            "selected_cpg_top1000.tsv",
            "baseline_result_notes.md",
        ],
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
