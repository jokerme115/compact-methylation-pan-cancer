from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from q1_feature_selection import select_features
from q1_metrics import calibration_metrics, multiclass_metrics, per_class_metrics, uncertainty_frame
from q1_train_ml_baselines import (
    fold_pairs,
    labels_for_rows,
    prepare_fold_matrix_mapping,
    resolve_matrix_rows,
    subset_rows,
)


ANNOTATION_COLUMNS = [
    "IlmnID",
    "CHR",
    "MAPINFO",
    "UCSC_RefGene_Name",
    "UCSC_RefGene_Group",
    "UCSC_CpG_Islands_Name",
    "Relation_to_UCSC_CpG_Island",
    "Regulatory_Feature_Name",
    "Regulatory_Feature_Group",
    "DHS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe LR small-CpG-panel experiments and export paper-ready "
            "CpG biomarker tables."
        )
    )
    parser.add_argument("--matrix-npy", required=True)
    parser.add_argument(
        "--matrix-samples-tsv",
        default="",
        help="Optional source manifest defining the row order of --matrix-npy.",
    )
    parser.add_argument("--probe-ids", required=True, help="Candidate probe ID TSV with column probe_id.")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--annotation-csv-gz", required=True, help="Illumina 450k GPL annotation csv.gz.")
    parser.add_argument("--pathway-tsv", default="", help="Optional Reactome pathway-gene TSV.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-method", default="variance")
    parser.add_argument("--panel-sizes", default="10,20,50,100,200,500,1000")
    parser.add_argument("--interpret-panel-size", type=int, default=100)
    parser.add_argument("--class-top-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--max-fold-runs", type=int, default=0, help="Debug limiter; 0 means all 15 folds.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise ValueError("--panel-sizes must contain positive integers")
    return values


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def load_probe_ids(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError("probe IDs file must contain column: probe_id")
    return df["probe_id"].astype(str).tolist()


def model_predict_proba(pipe: Pipeline, x: np.ndarray, n_classes: int) -> np.ndarray:
    proba = pipe.predict_proba(x)
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    classes = pipe.named_steps["model"].classes_.astype(int)
    full = np.zeros((x.shape[0], n_classes), dtype=float)
    full[:, classes] = proba
    row_sums = full.sum(axis=1, keepdims=True)
    full = np.divide(full, row_sums, out=np.full_like(full, 1.0 / n_classes), where=row_sums > 0)
    full = np.clip(full, 1e-12, 1.0)
    return full / full.sum(axis=1, keepdims=True)


def build_lr(seed: int, n_jobs: int, max_iter: int) -> Pipeline:
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
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def selection_needs_imputation(method: str) -> bool:
    return method.lower() in {"anova", "f_classif", "ftest", "mutual_info", "mi"}


def load_annotation(annotation_csv_gz: Path, selected_probe_ids: set[str]) -> pd.DataFrame:
    annot = pd.read_csv(
        annotation_csv_gz,
        compression="gzip",
        skiprows=7,
        usecols=lambda col: col in ANNOTATION_COLUMNS,
        low_memory=False,
    )
    annot = annot.rename(
        columns={
            "IlmnID": "probe_id",
            "CHR": "chr",
            "MAPINFO": "mapinfo",
            "UCSC_RefGene_Name": "gene_symbol_raw",
            "UCSC_RefGene_Group": "gene_region_raw",
            "UCSC_CpG_Islands_Name": "cpg_island",
            "Relation_to_UCSC_CpG_Island": "relation_to_cpg_island",
            "Regulatory_Feature_Name": "regulatory_feature",
            "Regulatory_Feature_Group": "regulatory_feature_group",
            "DHS": "dhs",
        }
    )
    annot = annot[annot["probe_id"].astype(str).isin(selected_probe_ids)].copy()
    for col in annot.columns:
        annot[col] = annot[col].fillna("").astype(str)
    annot["gene_symbol"] = annot["gene_symbol_raw"].str.split(";").str[0]
    annot["gene_region"] = annot["gene_region_raw"].str.split(";").str[0]
    annot["is_promoter"] = annot["gene_region_raw"].str.contains("TSS1500|TSS200|5'UTR|1stExon", regex=True, na=False)
    return annot


def load_gene_pathways(pathway_tsv: str) -> pd.DataFrame:
    if not pathway_tsv:
        return pd.DataFrame(columns=["gene_symbol", "pathway_count", "top_pathways"])
    path = Path(pathway_tsv)
    if not path.exists():
        return pd.DataFrame(columns=["gene_symbol", "pathway_count", "top_pathways"])
    pathways = pd.read_csv(path, sep="\t")
    if not {"gene_symbol", "pathway"}.issubset(pathways.columns):
        return pd.DataFrame(columns=["gene_symbol", "pathway_count", "top_pathways"])
    grouped = (
        pathways.dropna(subset=["gene_symbol", "pathway"])
        .assign(gene_symbol=lambda df: df["gene_symbol"].astype(str), pathway=lambda df: df["pathway"].astype(str))
        .groupby("gene_symbol")["pathway"]
        .agg(lambda vals: ";".join(sorted(set(vals))[:5]))
        .reset_index(name="top_pathways")
    )
    grouped["pathway_count"] = grouped["top_pathways"].str.count(";") + grouped["top_pathways"].ne("").astype(int)
    return grouped


def annotate_table(df: pd.DataFrame, annotation: pd.DataFrame, gene_pathways: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(annotation, on="probe_id", how="left")
    if not gene_pathways.empty:
        out = out.merge(gene_pathways, on="gene_symbol", how="left")
    if "pathway_count" in out.columns:
        out["pathway_count"] = out["pathway_count"].fillna(0).astype(int)
        out["top_pathways"] = out["top_pathways"].fillna("")
    return out


def write_summary_report(out_dir: Path, summary_df: pd.DataFrame, best_panel: pd.Series | None) -> None:
    lines = [
        "# LR Small CpG Panel Results",
        "",
        "This report is generated from leakage-safe patient-level repeated CV.",
        "",
        "## Main Result",
    ]
    if best_panel is not None:
        lines.extend(
            [
                f"- Best panel by mean Macro F1: top{int(best_panel['panel_size'])}.",
                f"- Macro F1: {best_panel['macro_f1_mean']:.4f} +/- {best_panel['macro_f1_std']:.4f}.",
                f"- Balanced accuracy: {best_panel['balanced_accuracy_mean']:.4f}.",
                f"- Accuracy: {best_panel['accuracy_mean']:.4f}.",
                f"- Top-3 accuracy: {best_panel.get('top3_accuracy_mean', float('nan')):.4f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Paper Interpretation",
            "",
            "- Use LR as the primary compact classifier.",
            "- Use PathMethNet as a complementary CpG-Gene-Pathway interpretability module.",
            "- Do not claim the deep model outperforms LR unless future results support it.",
            "",
            "## Key Output Files",
            "",
            "- `summary_metrics_by_panel.csv`: panel-size performance table.",
            "- `fold_metrics.tsv`: repeated-CV fold-level metrics.",
            "- `stable_cpg_panel_annotated.tsv`: stable CpG list across folds.",
            "- `class_specific_top_cpg_annotated.tsv`: per-cancer top CpG markers.",
            "- `lr_coefficients_interpret_panel.tsv`: LR coefficients for the interpretation panel.",
        ]
    )
    out_dir.joinpath("lr_small_panel_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = out_dir / "selected_features"
    coef_dir = out_dir / "coefficients"
    pred_dir = out_dir / "predictions"
    for path in (selected_dir, coef_dir, pred_dir):
        path.mkdir(parents=True, exist_ok=True)

    panel_sizes = parse_int_list(args.panel_sizes)
    max_panel_size = max(panel_sizes)
    if args.interpret_panel_size not in panel_sizes:
        raise ValueError("--interpret-panel-size must be one of --panel-sizes")

    splits_dir = Path(args.splits_dir)
    samples_df = pd.read_csv(splits_dir / "samples_locked.tsv", sep="\t")
    folds_df = pd.read_csv(splits_dir / "fold_assignments.tsv", sep="\t")
    split_summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    label_classes = list(split_summary["labels"])
    n_classes = len(label_classes)
    if "q1_label_id" not in samples_df.columns:
        label_to_id = {label: idx for idx, label in enumerate(label_classes)}
        samples_df["q1_label_id"] = samples_df["project_id"].astype(str).map(label_to_id)
        if samples_df["q1_label_id"].isna().any():
            raise ValueError("samples manifest contains project_id values absent from split_summary labels")
        samples_df["q1_label_id"] = samples_df["q1_label_id"].astype(int)

    probe_ids = load_probe_ids(Path(args.probe_ids))
    x = np.load(args.matrix_npy, mmap_mode="r")
    if x.shape[1] != len(probe_ids):
        raise ValueError(f"Probe ID count {len(probe_ids)} does not match matrix columns {x.shape[1]}")
    locked_to_matrix_rows = resolve_matrix_rows(samples_df, args.matrix_samples_tsv, x.shape[0])
    samples_df, fold_to_matrix = prepare_fold_matrix_mapping(samples_df, folds_df, locked_to_matrix_rows)

    fold_index = fold_pairs(folds_df)
    if args.max_fold_runs > 0:
        fold_index = fold_index[: args.max_fold_runs]

    fold_metrics_path = out_dir / "fold_metrics.tsv"
    completed_run_ids: set[str] = set()
    metric_rows: list[dict[str, Any]] = []
    if args.resume and fold_metrics_path.exists():
        old_metrics = pd.read_csv(fold_metrics_path, sep="\t")
        if "run_id" in old_metrics.columns:
            completed_run_ids = set(old_metrics["run_id"].astype(str))
            metric_rows.extend(old_metrics.to_dict("records"))
            log_event("resume_state_loaded", n_completed=len(completed_run_ids))

    class_metric_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    stable_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for run_number, (repeat, fold) in enumerate(fold_index, start=1):
        fold_start = time.time()
        log_event("fold_start", run_number=run_number, repeat=repeat, fold=fold)
        train_rows = subset_rows(folds_df, repeat, fold, "train")
        test_rows = subset_rows(folds_df, repeat, fold, "test")
        y_train = labels_for_rows(samples_df, train_rows)
        y_test = labels_for_rows(samples_df, test_rows)

        log_event("load_matrix_rows_start", repeat=repeat, fold=fold, n_train=len(train_rows), n_test=len(test_rows))
        x_train_raw = np.asarray(x[fold_to_matrix[train_rows]], dtype=np.float32)
        x_test_raw = np.asarray(x[fold_to_matrix[test_rows]], dtype=np.float32)
        log_event("load_matrix_rows_done", repeat=repeat, fold=fold, seconds=time.time() - fold_start)

        log_event("feature_selection_start", repeat=repeat, fold=fold, feature_method=args.feature_method, top_k=max_panel_size)
        if selection_needs_imputation(args.feature_method):
            selection_imputer = SimpleImputer(strategy="median")
            x_train_for_selection = selection_imputer.fit_transform(x_train_raw)
        else:
            x_train_for_selection = x_train_raw
        selected_max_idx = select_features(
            x_train_for_selection,
            y_train,
            method=args.feature_method,
            top_k=max_panel_size,
            seed=args.seed,
        )
        selected_max_probe_ids = [probe_ids[i] for i in selected_max_idx]
        selected_df = pd.DataFrame(
            {
                "repeat": repeat,
                "fold": fold,
                "rank": np.arange(1, len(selected_max_idx) + 1),
                "probe_index": selected_max_idx,
                "probe_id": selected_max_probe_ids,
            }
        )
        selected_df.to_csv(selected_dir / f"selected_top{max_panel_size}_r{repeat:02d}f{fold:02d}.tsv", sep="\t", index=False)
        for rank, (probe_index, probe_id) in enumerate(zip(selected_max_idx, selected_max_probe_ids), start=1):
            stable_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "rank": rank,
                    "probe_index": int(probe_index),
                    "probe_id": probe_id,
                }
            )
        log_event("feature_selection_done", repeat=repeat, fold=fold, seconds=time.time() - fold_start)

        for panel_size in panel_sizes:
            run_id = f"logreg_panel__{args.feature_method}__top{panel_size}__r{repeat:02d}f{fold:02d}"
            if run_id in completed_run_ids:
                log_event("run_skip_completed", run_id=run_id)
                continue
            selected_idx = selected_max_idx[:panel_size]
            selected_probe_ids = [probe_ids[i] for i in selected_idx]
            x_train = x_train_raw[:, selected_idx]
            x_test = x_test_raw[:, selected_idx]

            log_event("model_fit_start", run_id=run_id, n_features=panel_size)
            pipe = build_lr(args.seed + repeat * 100 + fold, args.n_jobs, args.max_iter)
            fit_start = time.time()
            pipe.fit(x_train, y_train)
            log_event("model_fit_done", run_id=run_id, seconds=time.time() - fit_start)

            y_proba = model_predict_proba(pipe, x_test, n_classes=n_classes)
            y_pred = y_proba.argmax(axis=1)
            metrics = multiclass_metrics(y_test, y_pred, y_proba)
            cal = calibration_metrics(y_test, y_proba)
            row = {
                "run_id": run_id,
                "model": "logreg",
                "feature_method": args.feature_method,
                "panel_size": panel_size,
                "repeat": repeat,
                "fold": fold,
                "n_train": int(len(train_rows)),
                "n_test": int(len(test_rows)),
                "seconds": float(time.time() - fit_start),
                **metrics,
                **cal,
            }
            metric_rows.append(row)

            class_df = per_class_metrics(y_test, y_pred, label_classes)
            class_df.insert(0, "run_id", run_id)
            class_df.insert(1, "panel_size", panel_size)
            class_df.insert(2, "repeat", repeat)
            class_df.insert(3, "fold", fold)
            class_metric_rows.append(class_df)

            if args.save_predictions:
                pred_df = uncertainty_frame(y_test, y_proba, label_classes)
                pred_df.insert(0, "matrix_row", test_rows)
                pred_df.insert(0, "run_id", run_id)
                prediction_rows.append(pred_df)

            if panel_size == args.interpret_panel_size:
                model = pipe.named_steps["model"]
                coef = np.asarray(model.coef_, dtype=float)
                for class_id, label in enumerate(label_classes):
                    class_coef = coef[class_id]
                    order = np.argsort(np.abs(class_coef))[::-1]
                    top_order = order[: max(args.class_top_n, min(panel_size, 100))]
                    for feature_rank, local_idx in enumerate(top_order, start=1):
                        coefficient_rows.append(
                            {
                                "repeat": repeat,
                                "fold": fold,
                                "class_id": class_id,
                                "label": label,
                                "feature_rank": feature_rank,
                                "panel_size": panel_size,
                                "probe_index": int(selected_idx[local_idx]),
                                "probe_id": selected_probe_ids[local_idx],
                                "coef": float(class_coef[local_idx]),
                                "abs_coef": float(abs(class_coef[local_idx])),
                            }
                        )
            log_event("run_done", **row)

        log_event("fold_done", repeat=repeat, fold=fold, seconds=time.time() - fold_start)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(fold_metrics_path, sep="\t", index=False)
    if not metrics_df.empty:
        metric_cols = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "top3_accuracy",
            "top5_accuracy",
            "log_loss",
            "ece_15_bins",
            "brier_multiclass",
            "mean_confidence",
        ]
        summary = metrics_df.groupby(["model", "feature_method", "panel_size"], as_index=False)[metric_cols].agg(["mean", "std"])
        summary.columns = [
            "_".join([str(part) for part in col if part])
            if isinstance(col, tuple)
            else str(col)
            for col in summary.columns
        ]
        summary = summary.reset_index(drop=True)
        summary.to_csv(out_dir / "summary_metrics_by_panel.csv", index=False)
    else:
        summary = pd.DataFrame()

    if class_metric_rows:
        pd.concat(class_metric_rows, ignore_index=True).to_csv(out_dir / "per_class_metrics.tsv", sep="\t", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(out_dir / "predictions.tsv", sep="\t", index=False)

    stable_df = pd.DataFrame(stable_rows)
    if not stable_df.empty:
        stable_counts = []
        for panel_size in panel_sizes:
            subset = stable_df[stable_df["rank"] <= panel_size]
            grouped = (
                subset.groupby(["probe_index", "probe_id"], as_index=False)
                .agg(selected_count=("probe_id", "size"), mean_rank=("rank", "mean"), median_rank=("rank", "median"))
                .sort_values(["selected_count", "mean_rank"], ascending=[False, True])
            )
            grouped.insert(0, "panel_size", panel_size)
            grouped["selected_fraction"] = grouped["selected_count"] / max(len(fold_index), 1)
            stable_counts.append(grouped)
        stable_panel = pd.concat(stable_counts, ignore_index=True)
    else:
        stable_panel = pd.DataFrame(columns=["panel_size", "probe_index", "probe_id", "selected_count", "selected_fraction"])

    coef_df = pd.DataFrame(coefficient_rows)
    selected_for_annotation = set(stable_panel["probe_id"].astype(str).tolist())
    if not coef_df.empty:
        selected_for_annotation.update(coef_df["probe_id"].astype(str).tolist())
    annotation = load_annotation(Path(args.annotation_csv_gz), selected_for_annotation)
    gene_pathways = load_gene_pathways(args.pathway_tsv)

    if not stable_panel.empty:
        annotate_table(stable_panel, annotation, gene_pathways).to_csv(
            out_dir / "stable_cpg_panel_annotated.tsv", sep="\t", index=False
        )
    if not coef_df.empty:
        coef_df.to_csv(out_dir / "lr_coefficients_interpret_panel.tsv", sep="\t", index=False)
        class_top = (
            coef_df.groupby(["label", "class_id", "probe_index", "probe_id"], as_index=False)
            .agg(
                mean_abs_coef=("abs_coef", "mean"),
                mean_coef=("coef", "mean"),
                selected_fold_count=("abs_coef", "size"),
                mean_feature_rank=("feature_rank", "mean"),
            )
            .sort_values(["label", "mean_abs_coef", "selected_fold_count"], ascending=[True, False, False])
        )
        class_top["class_rank"] = class_top.groupby("label")["mean_abs_coef"].rank(method="first", ascending=False).astype(int)
        class_top = class_top[class_top["class_rank"] <= args.class_top_n].copy()
        annotate_table(class_top, annotation, gene_pathways).to_csv(
            out_dir / "class_specific_top_cpg_annotated.tsv", sep="\t", index=False
        )

    best_panel = None
    if not summary.empty:
        best_idx = summary["macro_f1_mean"].astype(float).idxmax()
        best_panel = summary.loc[best_idx]
    write_summary_report(out_dir, summary, best_panel)

    manifest = {
        "matrix_npy": str(Path(args.matrix_npy).resolve()),
        "matrix_samples_tsv": str(Path(args.matrix_samples_tsv).resolve()) if args.matrix_samples_tsv else "",
        "probe_ids": str(Path(args.probe_ids).resolve()),
        "splits_dir": str(splits_dir.resolve()),
        "annotation_csv_gz": str(Path(args.annotation_csv_gz).resolve()),
        "pathway_tsv": str(Path(args.pathway_tsv).resolve()) if args.pathway_tsv else "",
        "feature_method": args.feature_method,
        "panel_sizes": panel_sizes,
        "interpret_panel_size": args.interpret_panel_size,
        "class_top_n": args.class_top_n,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "max_iter": args.max_iter,
        "n_completed_runs": int(len(metric_rows)),
        "outputs": [
            "fold_metrics.tsv",
            "summary_metrics_by_panel.csv",
            "stable_cpg_panel_annotated.tsv",
            "class_specific_top_cpg_annotated.tsv",
            "lr_coefficients_interpret_panel.tsv",
            "lr_small_panel_report.md",
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log_event("all_done", out_dir=str(out_dir), n_metric_rows=len(metric_rows))


if __name__ == "__main__":
    main()
