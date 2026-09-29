from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from q1_feature_selection import select_features
from q1_metrics import bootstrap_ci, calibration_metrics, multiclass_metrics, per_class_metrics, uncertainty_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe ML baselines on locked Q1 CV splits. Feature selection is fit inside each fold."
    )
    parser.add_argument("--matrix-npy", required=True)
    parser.add_argument(
        "--matrix-samples-tsv",
        default="",
        help="Optional source manifest defining the row order of --matrix-npy. "
        "When supplied, locked split rows are mapped by file_id before indexing the matrix.",
    )
    parser.add_argument("--splits-dir", required=True, help="Directory from q1_make_splits.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", default="logreg,elasticnet,linear_svm,random_forest,extra_trees")
    parser.add_argument("--feature-methods", default="variance,anova")
    parser.add_argument("--top-k", default="1000,5000,10000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-fold-runs", type=int, default=0, help="Debug limiter; 0 means run all.")
    parser.add_argument("--save-models", action="store_true", help="Persist fitted sklearn pipelines for each fold.")
    parser.add_argument("--bootstrap", type=int, default=300, help="Bootstrap samples per fold for quick CIs.")
    parser.add_argument("--resume", action="store_true", help="Skip run_ids already present in fold_metrics.tsv or --resume-log.")
    parser.add_argument("--resume-log", default="", help="Optional previous run log containing JSON run_done events.")
    return parser.parse_args()


def build_model(name: str, seed: int, n_jobs: int) -> Any:
    key = name.lower()
    if key == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced", solver="saga", n_jobs=n_jobs, random_state=seed)
    if key == "elasticnet":
        return LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if key == "linear_svm":
        return SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed)
    if key == "sgd_logloss":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if key == "rbf_svm":
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if key == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if key in {"hist_gradient_boosting", "hgb"}:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1e-4,
            early_stopping=True,
            random_state=seed,
        )
    if key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            early_stopping=True,
            max_iter=200,
            random_state=seed,
        )
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install xgboost to use --models xgboost") from exc
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install lightgbm to use --models lightgbm") from exc
        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model: {name}")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def fold_pairs(folds_df: pd.DataFrame) -> list[tuple[int, int]]:
    return sorted({(int(row.repeat), int(row.fold)) for row in folds_df.itertuples()})


def subset_rows(folds_df: pd.DataFrame, repeat: int, fold: int, split: str) -> np.ndarray:
    mask = (folds_df["repeat"] == repeat) & (folds_df["fold"] == fold) & (folds_df["split"] == split)
    return folds_df.loc[mask, "matrix_row"].to_numpy(dtype=int)


def resolve_matrix_rows(samples_df: pd.DataFrame, matrix_samples_tsv: str, n_matrix_rows: int) -> np.ndarray:
    """Map locked split row positions to the physical matrix row order."""
    if not matrix_samples_tsv:
        rows = np.arange(len(samples_df), dtype=int)
    else:
        source = pd.read_csv(matrix_samples_tsv, sep="\t")
        if "file_id" not in samples_df.columns or "file_id" not in source.columns:
            raise ValueError("Both locked and matrix source manifests must contain file_id for row mapping")
        if source["file_id"].duplicated().any():
            raise ValueError("Matrix source manifest contains duplicated file_id values")
        source_rows = pd.Series(np.arange(len(source), dtype=int), index=source["file_id"].astype(str))
        missing = sorted(set(samples_df["file_id"].astype(str)) - set(source_rows.index))
        if missing:
            raise ValueError(f"Locked manifest contains {len(missing)} file_id values absent from matrix source manifest")
        rows = samples_df["file_id"].astype(str).map(source_rows).to_numpy(dtype=int)
    if len(rows) != len(samples_df) or (len(rows) and int(rows.max()) >= n_matrix_rows):
        raise ValueError("Resolved matrix row map is incompatible with the matrix dimensions")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Resolved matrix row map is not one-to-one")
    return rows


def prepare_fold_matrix_mapping(
    samples_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    locked_to_matrix_rows: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Make fold row IDs usable for both labels and physical matrix indexing."""
    fold_values = folds_df["matrix_row"].to_numpy(dtype=int)
    if len(fold_values) == 0:
        raise ValueError("fold_assignments.tsv is empty")
    out = samples_df.copy()
    if int(fold_values.max()) >= len(samples_df):
        if set(np.unique(fold_values)) - set(locked_to_matrix_rows):
            raise ValueError("fold matrix rows are not covered by the locked-to-source row map")
        out["matrix_row"] = locked_to_matrix_rows
        fold_to_matrix = np.arange(int(fold_values.max()) + 1, dtype=int)
    else:
        out["matrix_row"] = np.arange(len(samples_df), dtype=int)
        fold_to_matrix = locked_to_matrix_rows
    return out, fold_to_matrix


def labels_for_rows(samples_df: pd.DataFrame, matrix_rows: np.ndarray) -> np.ndarray:
    if "q1_label_id" not in samples_df.columns:
        raise ValueError("samples manifest must contain q1_label_id before labels_for_rows is called")
    if "matrix_row" in samples_df.columns:
        labels = samples_df.set_index("matrix_row").loc[matrix_rows, "q1_label_id"].to_numpy(dtype=int)
    else:
        labels = samples_df.iloc[matrix_rows]["q1_label_id"].to_numpy(dtype=int)
    return labels


def log_event(message: str, **fields: Any) -> None:
    payload = {"event": message, **fields}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_rows_from_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if '"event": "run_done"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row.pop("event", None)
        if "run_id" in row:
            rows.append(row)
    return rows


def selection_needs_imputation(method: str) -> bool:
    method_key = method.lower()
    return method_key in {"anova", "f_classif", "ftest", "mutual_info", "mi"}


def model_predict_proba(pipe: Pipeline, x: np.ndarray, n_classes: int) -> np.ndarray:
    proba = pipe.predict_proba(x)
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    classes = pipe.named_steps["model"].classes_.astype(int)
    full = np.zeros((x.shape[0], n_classes), dtype=float)
    full[:, classes] = proba
    row_sums = full.sum(axis=1, keepdims=True)
    normalized = np.divide(full, row_sums, out=np.full_like(full, 1.0 / n_classes), where=row_sums > 0)
    normalized = np.clip(normalized, 1e-12, 1.0)
    normalized = normalized / normalized.sum(axis=1, keepdims=True)
    return normalized


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    predictions_dir = out_dir / "predictions"
    features_dir = out_dir / "selected_features"
    for path in (models_dir, predictions_dir, features_dir):
        path.mkdir(parents=True, exist_ok=True)

    splits_dir = Path(args.splits_dir)
    samples_df = pd.read_csv(splits_dir / "samples_locked.tsv", sep="\t")
    folds_df = pd.read_csv(splits_dir / "fold_assignments.tsv", sep="\t")
    summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    label_classes = list(summary["labels"])
    n_classes = len(label_classes)
    if "q1_label_id" not in samples_df.columns:
        label_to_id = {label: idx for idx, label in enumerate(label_classes)}
        samples_df["q1_label_id"] = samples_df["project_id"].astype(str).map(label_to_id)
        if samples_df["q1_label_id"].isna().any():
            raise ValueError("samples manifest contains project_id values absent from split_summary labels")
        samples_df["q1_label_id"] = samples_df["q1_label_id"].astype(int)

    x = np.load(args.matrix_npy, mmap_mode="r")
    models = parse_csv_list(args.models)
    feature_methods = parse_csv_list(args.feature_methods)
    top_k_values = parse_int_list(args.top_k)
    max_top_k = max(top_k_values)

    fold_index = fold_pairs(folds_df)
    selection_cache: dict[tuple[str, int, int], np.ndarray] = {}
    run_rows: list[dict[str, Any]] = []
    per_class_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    completed_run_ids: set[str] = set()
    if args.resume:
        if (out_dir / "fold_metrics.tsv").exists():
            existing_metrics = pd.read_csv(out_dir / "fold_metrics.tsv", sep="\t")
            if "run_id" in existing_metrics.columns:
                run_rows.extend(existing_metrics.to_dict("records"))
                completed_run_ids.update(existing_metrics["run_id"].astype(str))
        if (out_dir / "per_class_metrics.tsv").exists():
            per_class_rows.append(pd.read_csv(out_dir / "per_class_metrics.tsv", sep="\t"))
        if (out_dir / "calibration_metrics.tsv").exists():
            calibration_rows.extend(pd.read_csv(out_dir / "calibration_metrics.tsv", sep="\t").to_dict("records"))
        if (out_dir / "bootstrap_ci.tsv").exists():
            bootstrap_rows.append(pd.read_csv(out_dir / "bootstrap_ci.tsv", sep="\t"))
        if args.resume_log:
            for row in run_rows_from_log(Path(args.resume_log)):
                run_id = str(row["run_id"])
                if run_id not in completed_run_ids:
                    run_rows.append(row)
                    completed_run_ids.add(run_id)
        log_event("resume_state_loaded", n_completed=len(completed_run_ids))
    run_count = 0

    for model_name in models:
        for feature_method in feature_methods:
            for top_k in top_k_values:
                for repeat, fold in fold_index:
                    run_count += 1
                    if args.max_fold_runs and run_count > args.max_fold_runs:
                        break
                    run_id = f"{model_name}__{feature_method}__top{top_k}__r{repeat:02d}f{fold:02d}"
                    start = time.time()
                    log_event(
                        "run_start",
                        run_id=run_id,
                        model=model_name,
                        feature_method=feature_method,
                        top_k=int(top_k),
                        repeat=int(repeat),
                        fold=int(fold),
                    )
                    if run_id in completed_run_ids:
                        log_event("run_skip_completed", run_id=run_id)
                        continue

                    train_rows = subset_rows(folds_df, repeat, fold, "train")
                    test_rows = subset_rows(folds_df, repeat, fold, "test")
                    y_train = labels_for_rows(samples_df, train_rows)
                    y_test = labels_for_rows(samples_df, test_rows)

                    log_event("load_matrix_rows_start", run_id=run_id, n_train=int(len(train_rows)), n_test=int(len(test_rows)))
                    x_train_raw = np.asarray(x[train_rows], dtype=np.float32)
                    x_test_raw = np.asarray(x[test_rows], dtype=np.float32)
                    log_event("load_matrix_rows_done", run_id=run_id, seconds=float(time.time() - start))

                    cache_key = (feature_method.lower(), int(repeat), int(fold))
                    selected_max_idx = selection_cache.get(cache_key)
                    if selected_max_idx is None:
                        if selection_needs_imputation(feature_method):
                            log_event("selection_impute_start", run_id=run_id)
                            selector_imputer = SimpleImputer(strategy="median")
                            x_train_for_selection = selector_imputer.fit_transform(x_train_raw)
                            log_event("selection_impute_done", run_id=run_id, seconds=float(time.time() - start))
                        else:
                            x_train_for_selection = x_train_raw
                            log_event("selection_impute_skipped", run_id=run_id, reason=f"{feature_method} is NaN-aware")

                        log_event("feature_selection_start", run_id=run_id, requested_top_k=int(max_top_k))
                        selected_max_idx = select_features(
                            x_train_for_selection,
                            y_train,
                            method=feature_method,
                            top_k=max_top_k,
                            seed=args.seed,
                        )
                        selection_cache[cache_key] = selected_max_idx
                        log_event(
                            "feature_selection_done",
                            run_id=run_id,
                            n_selected=int(len(selected_max_idx)),
                            cached_for_top_k=int(max_top_k),
                            seconds=float(time.time() - start),
                        )
                    else:
                        log_event(
                            "feature_selection_cache_hit",
                            run_id=run_id,
                            cached_top_k=int(len(selected_max_idx)),
                            seconds=float(time.time() - start),
                        )

                    selected_idx = selected_max_idx[: min(int(top_k), len(selected_max_idx))]
                    np.save(features_dir / f"{run_id}.npy", selected_idx)
                    log_event("feature_subset_ready", run_id=run_id, n_selected=int(len(selected_idx)), seconds=float(time.time() - start))

                    x_train = x_train_raw[:, selected_idx]
                    x_test = x_test_raw[:, selected_idx]
                    pipe = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler(with_mean=False)),
                            ("model", build_model(model_name, seed=args.seed + repeat * 100 + fold, n_jobs=args.n_jobs)),
                        ]
                    )
                    log_event("model_fit_start", run_id=run_id)
                    pipe.fit(x_train, y_train)
                    log_event("model_fit_done", run_id=run_id, seconds=float(time.time() - start))

                    log_event("predict_metrics_start", run_id=run_id)
                    y_proba = model_predict_proba(pipe, x_test, n_classes=n_classes)
                    y_pred = y_proba.argmax(axis=1)

                    metrics = multiclass_metrics(y_test, y_pred, y_proba)
                    row = {
                        "run_id": run_id,
                        "model": model_name,
                        "feature_method": feature_method,
                        "top_k": int(top_k),
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "n_train": int(len(train_rows)),
                        "n_test": int(len(test_rows)),
                        "seconds": float(time.time() - start),
                        **metrics,
                    }
                    run_rows.append(row)

                    pred_df = uncertainty_frame(y_test, y_proba, label_classes)
                    pred_df.insert(0, "matrix_row", test_rows)
                    pred_df.insert(0, "run_id", run_id)
                    pred_df.to_csv(predictions_dir / f"{run_id}.tsv", sep="\t", index=False)
                    np.savez_compressed(predictions_dir / f"{run_id}_proba.npz", y_true=y_test, y_pred=y_pred, y_proba=y_proba)

                    class_df = per_class_metrics(y_test, y_pred, label_classes)
                    class_df.insert(0, "run_id", run_id)
                    class_df.insert(1, "model", model_name)
                    class_df.insert(2, "feature_method", feature_method)
                    class_df.insert(3, "top_k", int(top_k))
                    per_class_rows.append(class_df)

                    cal = calibration_metrics(y_test, y_proba)
                    calibration_rows.append({"run_id": run_id, "model": model_name, "feature_method": feature_method, "top_k": int(top_k), **cal})

                    if args.bootstrap > 0:
                        boot_df = bootstrap_ci(
                            y_test,
                            y_pred,
                            y_proba,
                            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "top5_accuracy"],
                            n_bootstrap=args.bootstrap,
                            seed=args.seed,
                        )
                        boot_df.insert(0, "run_id", run_id)
                        bootstrap_rows.append(boot_df)

                    if args.save_models:
                        joblib.dump(pipe, models_dir / f"{run_id}.joblib")
                    log_event("run_done", **row)
                if args.max_fold_runs and run_count > args.max_fold_runs:
                    break
            if args.max_fold_runs and run_count > args.max_fold_runs:
                break
        if args.max_fold_runs and run_count > args.max_fold_runs:
            break

    metrics_df = pd.DataFrame(run_rows)
    metrics_df.to_csv(out_dir / "fold_metrics.tsv", sep="\t", index=False)
    if not metrics_df.empty:
        numeric_cols = [col for col in metrics_df.columns if col not in {"run_id", "model", "feature_method"}]
        grouped = metrics_df.groupby(["model", "feature_method", "top_k"], dropna=False)[numeric_cols].agg(["mean", "std"])
        grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
        grouped = grouped.reset_index()
        grouped.to_csv(out_dir / "summary_metrics_by_setting.csv", index=False)
    if per_class_rows:
        pd.concat(per_class_rows, ignore_index=True).to_csv(out_dir / "per_class_metrics.tsv", sep="\t", index=False)
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(out_dir / "calibration_metrics.tsv", sep="\t", index=False)
    if bootstrap_rows:
        pd.concat(bootstrap_rows, ignore_index=True).to_csv(out_dir / "bootstrap_ci.tsv", sep="\t", index=False)

    manifest = {
        "matrix_npy": str(Path(args.matrix_npy).resolve()),
        "splits_dir": str(splits_dir.resolve()),
        "models": models,
        "feature_methods": feature_methods,
        "top_k": top_k_values,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "bootstrap": args.bootstrap,
        "outputs": {
            "fold_metrics": "fold_metrics.tsv",
            "summary_metrics_by_setting": "summary_metrics_by_setting.csv",
            "per_class_metrics": "per_class_metrics.tsv",
            "calibration_metrics": "calibration_metrics.tsv",
            "bootstrap_ci": "bootstrap_ci.tsv",
            "predictions": "predictions/",
            "selected_features": "selected_features/",
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
