"""E6: paired bootstrap of the redundancy-aware arm against the variance arm.

Same patient-clustered, repeat-aware protocol as ``run_revision_e2_paired_bootstrap.py``
so that the E6 comparison is directly readable next to the E2 one.  Reads the
patient-level OOF files written by :mod:`run_revision_e6_redundancy_selection`
(``patient_oof_predictions_by_repeat.tsv.gz``), which are already aggregated by
probability-mean + argmax.

The two arms see exactly the same folds and the same panel sizes, so the
comparison is paired on ``(panel_size, repeat, q1_patient_id)``.

Output: ``paired_bootstrap_dedup_minus_variance.tsv`` plus a manifest.
Exploratory revision-stage evidence only -- ``fraction_gt_zero`` is not a p-value.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E6 paired bootstrap: dedup - variance_full.")
    p.add_argument("--arm-dir-a", required=True, help="baseline arm dir (variance_full)")
    p.add_argument("--arm-dir-b", required=True, help="alternative arm dir (dedup)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--panel-sizes", default="200,500,1000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


def patient_metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, recall_score

    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_acc": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "accuracy": float((yt == yp).mean()),
    }


def read_arm(arm_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        arm_dir / "patient_oof_predictions_by_repeat.tsv.gz", sep="\t", compression="gzip"
    )
    df["q1_patient_id"] = df["q1_patient_id"].astype(str)
    return df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    panel_sizes = [int(v) for v in args.panel_sizes.split(",") if v.strip()]
    a_all = read_arm(Path(args.arm_dir_a))
    b_all = read_arm(Path(args.arm_dir_b))

    rows: list[dict] = []
    for panel in panel_sizes:
        a = a_all[a_all["panel_size"] == panel].copy()
        b = b_all[b_all["panel_size"] == panel].copy()
        merged = a.merge(
            b,
            on=["panel_size", "repeat", "q1_patient_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        n_repeats = merged["repeat"].nunique()
        n_patients = merged["q1_patient_id"].nunique()
        if len(merged) != n_repeats * n_patients:
            raise RuntimeError(
                f"panel={panel}: expected {n_repeats} x {n_patients} rows, got {len(merged)}"
            )
        if not (merged["y_true_a"] == merged["y_true_b"]).all():
            raise RuntimeError(f"panel={panel}: y_true mismatch between arms")

        patients = sorted(merged["q1_patient_id"].unique())
        repeats = sorted(merged["repeat"].unique())
        yt = np.zeros((n_repeats, n_patients), dtype=int)
        yp_a = np.zeros((n_repeats, n_patients), dtype=int)
        yp_b = np.zeros((n_repeats, n_patients), dtype=int)
        for ri, rep in enumerate(repeats):
            block = (
                merged[merged["repeat"] == rep].set_index("q1_patient_id").loc[patients]
            )
            yt[ri] = block["y_true_a"].to_numpy()
            yp_a[ri] = block["y_pred_a"].to_numpy()
            yp_b[ri] = block["y_pred_b"].to_numpy()

        point = {"macro_f1": [], "balanced_acc": [], "accuracy": []}
        abs_a = {"macro_f1": [], "balanced_acc": [], "accuracy": []}
        abs_b = {"macro_f1": [], "balanced_acc": [], "accuracy": []}
        for ri in range(n_repeats):
            ma = patient_metrics(yt[ri], yp_a[ri])
            mb = patient_metrics(yt[ri], yp_b[ri])
            for key in point:
                point[key].append(mb[key] - ma[key])
                abs_a[key].append(ma[key])
                abs_b[key].append(mb[key])

        boot = {k: np.empty(args.n_bootstrap) for k in point}
        for bi in range(args.n_bootstrap):
            idx = rng.integers(0, n_patients, size=n_patients)
            acc = {k: np.zeros(n_repeats) for k in point}
            for ri in range(n_repeats):
                ma = patient_metrics(yt[ri][idx], yp_a[ri][idx])
                mb = patient_metrics(yt[ri][idx], yp_b[ri][idx])
                for key in point:
                    acc[key][ri] = mb[key] - ma[key]
            for key in point:
                boot[key][bi] = float(np.mean(acc[key]))

        row = {
            "panel_size": panel,
            "n_patients_common": n_patients,
            "n_repeats": n_repeats,
            "a_macro_f1": round(float(np.mean(abs_a["macro_f1"])), 4),
            "b_macro_f1": round(float(np.mean(abs_b["macro_f1"])), 4),
            "a_balanced_acc": round(float(np.mean(abs_a["balanced_acc"])), 4),
            "b_balanced_acc": round(float(np.mean(abs_b["balanced_acc"])), 4),
            "a_accuracy": round(float(np.mean(abs_a["accuracy"])), 4),
            "b_accuracy": round(float(np.mean(abs_b["accuracy"])), 4),
        }
        for key in point:
            row[f"delta_{key}"] = round(float(np.mean(point[key])), 4)
            row[f"delta_{key}_ci_low"] = round(float(np.percentile(boot[key], 2.5)), 4)
            row[f"delta_{key}_ci_high"] = round(float(np.percentile(boot[key], 97.5)), 4)
            row[f"fraction_delta_{key}_gt_zero"] = round(float((boot[key] > 0).mean()), 4)
        rows.append(row)
        print(f"[panel {panel}] delta_macro_f1={row['delta_macro_f1']:+.4f} "
              f"CI=[{row['delta_macro_f1_ci_low']:+.4f},{row['delta_macro_f1_ci_high']:+.4f}]",
              flush=True)

    df = pd.DataFrame(rows)
    tmp_path = out_dir / ".paired_bootstrap_dedup_minus_variance.tsv.tmp"
    df.to_csv(tmp_path, sep="\t", index=False)
    os.replace(str(tmp_path), str(out_dir / "paired_bootstrap_dedup_minus_variance.tsv"))

    manifest = {
        "experiment": "E6_paired_bootstrap_dedup_minus_variance",
        "estimator": "mean of repeat-specific patient-level metric deltas",
        "bootstrap": "patient-clustered, patient draw applied to all repeats simultaneously",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "arm_a": args.arm_dir_a,
        "arm_b": args.arm_dir_b,
        "panel_sizes": panel_sizes,
        "note": "delta = dedup - variance_full; exploratory revision-stage comparison; "
                "fraction_delta_*_gt_zero is not a p-value",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[E6paired] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
