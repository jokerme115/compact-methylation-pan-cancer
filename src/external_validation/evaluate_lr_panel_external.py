from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCGA LR CpG panels and evaluate on curated external cohorts.")
    parser.add_argument("--train-matrix-npy", required=True)
    parser.add_argument("--train-samples-tsv", required=True)
    parser.add_argument("--source-samples-tsv", default=None, help="Optional pre-filter manifest used to audit excluded normal rows.")
    parser.add_argument("--train-probe-ids", required=True)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--stable-panel-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--panel-sizes", default="100,200,500,1000")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def load_probe_ids(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError(f"{path} must contain probe_id")
    return df["probe_id"].astype(str).tolist()


def build_lr(max_iter: int, n_jobs: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=max_iter,
                    n_jobs=n_jobs,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    top = np.argsort(proba, axis=1)[:, -k:]
    pred_sets = classes[top]
    return float(np.mean([y_true[i] in pred_sets[i] for i in range(len(y_true))]))


def parse_accepted_labels(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(";") if item.strip()}


def accepted_topk_accuracy(accepted: list[set[str]], proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    top = np.argsort(proba, axis=1)[:, -k:]
    pred_sets = classes[top]
    return float(np.mean([bool(accepted[i].intersection(pred_sets[i])) for i in range(len(accepted))]))


def confidence_curve(correct: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        keep = confidence >= threshold
        rows.append(
            {
                "threshold": threshold,
                "coverage": float(keep.mean()),
                "n_retained": int(keep.sum()),
                "accuracy_retained": float(correct[keep].mean()) if keep.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_sizes = [int(item) for item in args.panel_sizes.split(",") if item.strip()]
    train_probe_ids = load_probe_ids(Path(args.train_probe_ids))
    train_probe_to_idx = {probe: i for i, probe in enumerate(train_probe_ids)}
    ext_dir = Path(args.external_dir)
    external_probe_ids = load_probe_ids(ext_dir / "probe_ids.tsv")
    external_probe_to_idx = {probe: i for i, probe in enumerate(external_probe_ids)}

    train_samples = pd.read_csv(args.train_samples_tsv, sep="\t")
    if "project_id" not in train_samples.columns:
        raise ValueError("train samples must contain project_id")

    x_train_all = np.load(args.train_matrix_npy, mmap_mode="r")
    x_external_all = np.load(ext_dir / "X_external.npy", mmap_mode="r")
    if x_train_all.shape[0] != len(train_samples):
        raise ValueError(
            f"Training matrix/sample manifest mismatch: matrix has {x_train_all.shape[0]} rows, "
            f"manifest has {len(train_samples)} rows"
        )
    if "tissue_type" not in train_samples.columns:
        raise ValueError("train samples must contain tissue_type so the formal tumor-only cohort is auditable")
    tumor_mask = train_samples["tissue_type"].astype(str).str.lower().eq("tumor").to_numpy()
    if int(tumor_mask.sum()) != 9065:
        raise ValueError(f"Expected 9065 tumor rows in the formal training cohort, found {int(tumor_mask.sum())}")
    n_excluded_normal = int((~tumor_mask).sum())
    source_n_excluded_normal = n_excluded_normal
    if args.source_samples_tsv:
        source_samples = pd.read_csv(args.source_samples_tsv, sep="\t")
        if "tissue_type" not in source_samples.columns:
            raise ValueError("source samples manifest must contain tissue_type")
        source_n_excluded_normal = int(
            source_samples["tissue_type"].astype(str).str.lower().ne("tumor").sum()
        )
    train_samples = train_samples.loc[tumor_mask].reset_index(drop=True)
    y_train_labels = train_samples["project_id"].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    classes = label_encoder.classes_

    external_samples = pd.read_csv(ext_dir / "samples_external.tsv", sep="\t")
    y_external_labels = external_samples["label"].astype(str).to_numpy()
    if "accepted_labels" in external_samples.columns:
        accepted_labels = external_samples["accepted_labels"].astype(str).map(parse_accepted_labels).tolist()
    else:
        accepted_labels = [{label} for label in y_external_labels]
    exact_labels_available = set(y_external_labels).issubset(set(classes))
    panel_df = pd.read_csv(args.stable_panel_tsv, sep="\t")
    if not {"panel_size", "probe_id"}.issubset(panel_df.columns):
        raise ValueError("stable panel TSV must contain panel_size and probe_id")

    metric_rows = []
    prediction_frames = []
    curve_frames = []
    for panel_size in panel_sizes:
        panel_probe_ids = panel_df.loc[panel_df["panel_size"].eq(panel_size), "probe_id"].astype(str).tolist()
        if not panel_probe_ids:
            raise ValueError(f"No probes found for panel_size={panel_size}")
        usable_probe_ids = [
            probe for probe in panel_probe_ids if probe in train_probe_to_idx and probe in external_probe_to_idx
        ]
        if len(usable_probe_ids) < panel_size:
            print(
                json.dumps(
                    {
                        "event": "panel_probe_drop",
                        "panel_size": panel_size,
                        "requested": len(panel_probe_ids),
                        "usable": len(usable_probe_ids),
                    }
                ),
                flush=True,
            )
        train_idx = [train_probe_to_idx[probe] for probe in usable_probe_ids]
        ext_idx = [external_probe_to_idx[probe] for probe in usable_probe_ids]

        # Select the small panel before applying the row mask.  Applying the
        # boolean mask first would materialise the full candidate matrix.
        x_train = np.asarray(x_train_all[:, train_idx], dtype=np.float32)[tumor_mask]
        x_external = np.asarray(x_external_all[:, ext_idx], dtype=np.float32)
        model = build_lr(args.max_iter, args.n_jobs)
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_external)
        pred_int = model.predict(x_external)
        pred_labels = label_encoder.inverse_transform(pred_int)
        confidence = proba.max(axis=1)
        accepted_correct = np.array([pred_labels[i] in accepted_labels[i] for i in range(len(pred_labels))], dtype=bool)
        exact_correct = pred_labels == y_external_labels if exact_labels_available else np.zeros_like(accepted_correct)

        exact_accuracy = accuracy_score(y_external_labels, pred_labels) if exact_labels_available else np.nan
        exact_macro_f1 = (
            f1_score(y_external_labels, pred_labels, labels=sorted(set(y_external_labels)), average="macro")
            if exact_labels_available
            else np.nan
        )
        exact_bacc = balanced_accuracy_score(y_external_labels, pred_labels) if exact_labels_available else np.nan
        exact_top3 = topk_accuracy(y_external_labels, proba, classes, 3) if exact_labels_available else np.nan

        metric_rows.append(
            {
                "panel_size": panel_size,
                "n_panel_probes_requested": len(panel_probe_ids),
                "n_panel_probes_used": len(usable_probe_ids),
                "n_external_samples": len(y_external_labels),
                "external_labels": ";".join(sorted(set(y_external_labels))),
                "exact_accuracy": exact_accuracy,
                "exact_macro_f1_present_labels": exact_macro_f1,
                "exact_balanced_accuracy": exact_bacc,
                "exact_top3_accuracy": exact_top3,
                "accepted_top1_accuracy": float(accepted_correct.mean()),
                "accepted_top3_accuracy": accepted_topk_accuracy(accepted_labels, proba, classes, 3),
                "mean_confidence": float(np.mean(confidence)),
            }
        )

        top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
        pred_df = external_samples.copy()
        pred_df["panel_size"] = panel_size
        pred_df["pred_label"] = pred_labels
        pred_df["exact_correct"] = exact_correct
        pred_df["accepted_correct"] = accepted_correct
        pred_df["confidence"] = confidence
        for rank in range(3):
            pred_df[f"top{rank + 1}_label"] = classes[top3_idx[:, rank]]
            pred_df[f"top{rank + 1}_prob"] = proba[np.arange(proba.shape[0]), top3_idx[:, rank]]
        prediction_frames.append(pred_df)

        curve = confidence_curve(accepted_correct, confidence)
        curve.insert(0, "panel_size", panel_size)
        curve_frames.append(curve)

        cm_labels = sorted(set(y_external_labels) | set(pred_labels))
        cm = pd.DataFrame(confusion_matrix(y_external_labels, pred_labels, labels=cm_labels), index=cm_labels, columns=cm_labels)
        cm.to_csv(out_dir / f"confusion_matrix_panel{panel_size}.csv")
        if args.save_models:
            joblib.dump(model, out_dir / f"lr_panel{panel_size}.joblib")

        print(json.dumps({"event": "panel_done", "panel_size": panel_size}, ensure_ascii=False), flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "external_panel_metrics.tsv", sep="\t", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out_dir / "external_predictions.tsv", sep="\t", index=False)
    pd.concat(curve_frames, ignore_index=True).to_csv(out_dir / "external_confidence_curve.tsv", sep="\t", index=False)
    (out_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "external_dir": str(ext_dir.resolve()),
                "train_matrix": str(Path(args.train_matrix_npy).resolve()),
                "train_samples_manifest": str(Path(args.train_samples_tsv).resolve()),
                "train_source_manifest": str(Path(args.source_samples_tsv).resolve()) if args.source_samples_tsv else None,
                "train_tissue_filter": "tissue_type == tumor",
                "panel_sizes": panel_sizes,
                "n_train_samples": int(tumor_mask.sum()),
                "n_train_tumor": int(tumor_mask.sum()),
                "n_train_excluded_normal": source_n_excluded_normal,
                "n_train_classes": int(len(classes)),
                "n_external_samples": int(len(y_external_labels)),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"event": "all_done", "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
