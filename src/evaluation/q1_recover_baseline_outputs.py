from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from q1_metrics import bootstrap_ci, calibration_metrics, per_class_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover baseline metric tables from a q1_train_ml_baselines JSON log.")
    parser.add_argument("--log", required=True, help="Log file containing JSON run_done events.")
    parser.add_argument("--out-dir", required=True, help="Baseline output directory containing predictions/*.npz.")
    parser.add_argument("--splits-dir", required=True, help="Directory from q1_make_splits.py, used for label names.")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_rows_from_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if '"event": "run_done"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        row.pop("event", None)
        run_id = str(row.get("run_id", ""))
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        rows.append(row)
    return rows


def write_summary(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    metrics_df.to_csv(out_dir / "fold_metrics.tsv", sep="\t", index=False)
    if metrics_df.empty:
        return
    numeric_cols = [col for col in metrics_df.columns if col not in {"run_id", "model", "feature_method"}]
    grouped = metrics_df.groupby(["model", "feature_method", "top_k"], dropna=False)[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped.reset_index().to_csv(out_dir / "summary_metrics_by_setting.csv", index=False)


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    predictions_dir = out_dir / "predictions"
    splits_dir = Path(args.splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    label_classes = list(summary["labels"])

    rows = run_rows_from_log(log_path)
    metrics_df = pd.DataFrame(rows)
    write_summary(metrics_df, out_dir)

    per_class_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    recovered_npz = 0
    missing_npz: list[str] = []

    for row in rows:
        run_id = str(row["run_id"])
        npz_path = predictions_dir / f"{run_id}_proba.npz"
        if not npz_path.exists():
            missing_npz.append(run_id)
            continue
        data = np.load(npz_path)
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        y_proba = data["y_proba"]
        recovered_npz += 1

        class_df = per_class_metrics(y_true, y_pred, label_classes)
        class_df.insert(0, "run_id", run_id)
        class_df.insert(1, "model", row["model"])
        class_df.insert(2, "feature_method", row["feature_method"])
        class_df.insert(3, "top_k", int(row["top_k"]))
        per_class_rows.append(class_df)

        cal = calibration_metrics(y_true, y_proba)
        calibration_rows.append(
            {
                "run_id": run_id,
                "model": row["model"],
                "feature_method": row["feature_method"],
                "top_k": int(row["top_k"]),
                **cal,
            }
        )

        if args.bootstrap > 0:
            boot_df = bootstrap_ci(
                y_true,
                y_pred,
                y_proba,
                ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "top5_accuracy"],
                n_bootstrap=args.bootstrap,
                seed=args.seed,
            )
            boot_df.insert(0, "run_id", run_id)
            bootstrap_rows.append(boot_df)

    if per_class_rows:
        pd.concat(per_class_rows, ignore_index=True).to_csv(out_dir / "per_class_metrics.tsv", sep="\t", index=False)
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(out_dir / "calibration_metrics.tsv", sep="\t", index=False)
    if bootstrap_rows:
        pd.concat(bootstrap_rows, ignore_index=True).to_csv(out_dir / "bootstrap_ci.tsv", sep="\t", index=False)

    manifest = {
        "log": str(log_path.resolve()),
        "out_dir": str(out_dir.resolve()),
        "splits_dir": str(splits_dir.resolve()),
        "n_run_done": len(rows),
        "n_prediction_npz_recovered": recovered_npz,
        "n_missing_prediction_npz": len(missing_npz),
        "missing_prediction_npz": missing_npz[:50],
        "outputs": {
            "fold_metrics": "fold_metrics.tsv",
            "summary_metrics_by_setting": "summary_metrics_by_setting.csv",
            "per_class_metrics": "per_class_metrics.tsv",
            "calibration_metrics": "calibration_metrics.tsv",
            "bootstrap_ci": "bootstrap_ci.tsv",
        },
    }
    (out_dir / "recovery_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
