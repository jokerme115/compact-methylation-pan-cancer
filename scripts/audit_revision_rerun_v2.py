from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROB_E2 = "probability_class_"
PROB_E5 = "probability__"
E3_NAME = re.compile(
    r"^(?P<model>hierarchical|xgboost)__(?P<selection>\w+)__top"
    r"(?P<panel>\d+)__r(?P<repeat>\d+)f(?P<fold>\d+)\.tsv$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def verify_manifest(root: Path) -> dict:
    manifest_path = root / "total_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    missing: list[str] = []
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    for rel, expected in manifest.items():
        path = root / rel
        if len(str(path)) >= 248:
            path = Path("\\\\?\\" + str(path.resolve()))
        if not path.is_file():
            missing.append(rel)
            continue
        if path.stat().st_size != int(expected["size"]):
            size_mismatches.append(rel)
            continue
        if sha256(path) != expected["sha256"]:
            hash_mismatches.append(rel)
    # rglob under a non-long-path-aware Python can silently omit >260-char
    # members, so manifest verification is authoritative for the file count.
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "total_manifest.json"
    }
    return {
        "manifest_entries": len(manifest),
        "actual_files_visible_to_rglob": len(actual),
        "missing": missing,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "unlisted": sorted(actual - set(manifest)),
    }


def audit_e2(root: Path, probe_ids: np.ndarray) -> tuple[dict, pd.DataFrame]:
    run = root / "E2_feature_selection" / "run_v2"
    gate = pd.read_csv(run / "convergence_gate.tsv", sep="\t")
    convergence = pd.read_csv(run / "fold_convergence.tsv", sep="\t")
    manifest = pd.read_csv(run / "selected_probe_manifest.tsv", sep="\t")
    valid = manifest["probe_index"].between(0, len(probe_ids) - 1)
    mapping_match = np.zeros(len(manifest), dtype=bool)
    mapping_match[valid] = (
        probe_ids[manifest.loc[valid, "probe_index"].astype(int).to_numpy()]
        == manifest.loc[valid, "probe_id"].astype(str).to_numpy()
    )

    prediction_rows: list[dict] = []
    method_tables: dict[str, pd.DataFrame] = {}
    for method_dir in sorted(p for p in run.iterdir() if p.is_dir()):
        pred_path = method_dir / "patient_oof_predictions_by_repeat.tsv.gz"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path, sep="\t")
        method_tables[method_dir.name] = pred
        prob_cols = sorted(
            [c for c in pred if c.startswith(PROB_E2)],
            key=lambda c: int(c.rsplit("_", 1)[1]),
        )
        for (panel, repeat), frame in pred.groupby(["panel_size", "repeat"], sort=True):
            probabilities = frame[prob_cols].to_numpy(float)
            y_true = frame["y_true"].to_numpy(int)
            y_pred = probabilities.argmax(axis=1)
            prediction_rows.append(
                {
                    "method": method_dir.name,
                    "panel_size": int(panel),
                    "repeat": int(repeat),
                    "n_rows": len(frame),
                    "n_patients": frame["q1_patient_id"].nunique(),
                    "duplicate_patient_rows": int(frame.duplicated("q1_patient_id").sum()),
                    "max_probability_sum_error": float(np.max(np.abs(probabilities.sum(1) - 1))),
                    "stored_prediction_mismatches": int((y_pred != frame["y_pred"].to_numpy(int)).sum()),
                    **table_metrics(y_true, y_pred),
                }
            )

    copied_method_outputs_identical = False
    if method_tables:
        first = next(iter(method_tables.values()))
        copied_method_outputs_identical = all(first.equals(frame) for frame in method_tables.values())

    # The v2 runner passed the all-method OOF file to the evaluator six times
    # without filtering run_id.  Recover the actual method-specific estimates
    # directly from that combined OOF artifact.
    combined = pd.read_csv(run / "oof_sample_predictions.tsv.gz", sep="\t")
    combined["method"] = combined["run_id"].astype(str).str.split("__").str[0]
    combined_prob_cols = sorted(
        [c for c in combined if c.startswith(PROB_E2)],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    recovered_rows: list[dict] = []
    for (method, panel, repeat), frame in combined.groupby(
        ["method", "panel_size", "repeat"], sort=True
    ):
        patient_prob = frame.groupby("q1_patient_id", sort=True)[combined_prob_cols].mean()
        patient_true = frame.groupby("q1_patient_id", sort=True)["y_true"].agg(
            lambda values: int(values.mode().iloc[0])
        )
        probabilities = patient_prob.to_numpy(float)
        y_true = patient_true.loc[patient_prob.index].to_numpy(int)
        recovered_rows.append(
            {
                "method": method,
                "panel_size": int(panel),
                "repeat": int(repeat),
                "n_patients": len(patient_prob),
                "max_probability_sum_error": float(np.max(np.abs(probabilities.sum(1) - 1))),
                **table_metrics(y_true, probabilities.argmax(1)),
            }
        )
    recovered = pd.DataFrame(recovered_rows)
    recovered_summary = recovered.groupby(["method", "panel_size"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
    )
    top_level = pd.read_csv(
        root / "E2_feature_selection" / "e2_cross_method_primary_metrics_wide.tsv", sep="\t"
    )
    compare = recovered_summary.merge(
        top_level[["method", "panel_size", "accuracy", "balanced_accuracy", "macro_f1"]],
        on=["method", "panel_size"],
        suffixes=("_recovered", "_published_table"),
        validate="one_to_one",
    )
    aggregate_disagreement_rows = int(
        (
            np.abs(compare["macro_f1_recovered"] - compare["macro_f1_published_table"])
            > 5e-4
        ).sum()
    )

    stability = convergence[convergence["model_type"] == "saga_l1_subsamples"]
    result = {
        "gate_rows": len(gate),
        "gate_status_counts": gate["gate_status"].value_counts(dropna=False).to_dict(),
        "gate_nonconverged": int((~gate["converged"].astype(bool)).sum()),
        "gate_hit_max_iter": int(gate["hit_max_iter"].astype(bool).sum()),
        "fold_convergence_rows": len(convergence),
        "fold_nonconverged": int((~convergence["converged"].astype(bool)).sum()),
        "stability_min_subsample_converged_fraction": float(stability["subsample_converged_frac"].min()),
        "selected_manifest_rows": len(manifest),
        "selected_global_mapping_fraction": float(mapping_match.mean()),
        "boruta_note": sorted(
            convergence.loc[convergence["model_type"] == "BorutaPy_RF", "note"].dropna().unique().tolist()
        ),
        "combined_oof_rows": len(combined),
        "combined_oof_method_counts": combined["method"].value_counts().to_dict(),
        "per_method_evaluation_artifacts_identical": copied_method_outputs_identical,
        "per_method_evaluator_bug": (
            "run_revision_e2_convergence_rerun.py calls evaluate_repeated_oof on the same "
            "unfiltered all-method OOF file for every method directory."
        ),
        "top_level_aggregate_macro_f1_disagreement_rows": aggregate_disagreement_rows,
    }
    prediction_audit = pd.DataFrame(prediction_rows)
    prediction_audit.attrs["recovered"] = recovered
    prediction_audit.attrs["recovered_summary"] = recovered_summary
    prediction_audit.attrs["aggregate_compare"] = compare
    return result, prediction_audit


def majority_vote(frame: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    return (
        frame.groupby("q1_patient_id")
        .agg(
            y_true=(true_col, lambda values: values.mode().iloc[0]),
            y_pred=(pred_col, lambda values: values.mode().iloc[0]),
        )
        .reset_index()
    )


def audit_e3(root: Path, workspace: Path) -> tuple[dict, pd.DataFrame]:
    e3 = root / "E3_classifier_benchmark"
    run_v2 = e3 / "run_v2"
    locked_path = workspace / "data" / "tcga_450k" / "samples_locked.tsv"
    samples_used_path = workspace / "data" / "tcga_450k" / "samples_used.tsv"
    samples_used = pd.read_csv(samples_used_path, sep="\t").reset_index(drop=True)
    split_summary = json.loads(
        (workspace / "data" / "tcga_450k" / "split_summary.json").read_text(encoding="utf-8")
    )
    labels = list(split_summary["labels"])
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    # E3 prediction matrix_row values are physical positions in the 9,812-row
    # samples_used table.  This independently reconstructs the enriched
    # samples_locked mapping named by the server-side v2 script but omitted
    # from the archive.
    patient_core = samples_used["submitter_id"].astype(str).str.slice(0, 12)
    row_to_patient = (
        "tcga_participant::" + samples_used["project_id"].astype(str) + "::" + patient_core
    ).to_dict()
    row_to_label = samples_used["project_id"].astype(str).to_dict()

    raw_frames: list[pd.DataFrame] = []
    raw_mapping_mismatches = 0
    raw_unmapped_rows = 0
    for path in sorted((e3 / "run").glob("*/predictions/*.tsv")):
        match = E3_NAME.match(path.name)
        if not match:
            continue
        frame = pd.read_csv(path, sep="\t")
        mapped_patient = frame["matrix_row"].map(row_to_patient)
        mapped_label = frame["matrix_row"].map(row_to_label)
        raw_unmapped_rows += int(mapped_patient.isna().sum())
        raw_mapping_mismatches += int(
            (mapped_label.fillna("__UNMAPPED__").astype(str) != frame["true_label"].astype(str)).sum()
        )
        frame["q1_patient_id"] = mapped_patient
        frame["y_true_id"] = frame["true_label"].map(label_to_idx)
        frame["pred_id"] = frame["pred_label"].map(label_to_idx)
        frame["model"] = "hierarchical" if match["model"] == "hierarchical" else "xgboost_baseline"
        frame["selection"] = match["selection"]
        frame["panel_size"] = int(match["panel"])
        raw_frames.append(frame.dropna(subset=["q1_patient_id"]))
    raw = pd.concat(raw_frames, ignore_index=True)

    comparisons: list[dict] = []
    stored_mismatch_total = 0
    for (model, selection, panel), frame in raw.groupby(["model", "selection", "panel_size"], sort=True):
        reproduced = majority_vote(frame, "y_true_id", "pred_id").sort_values("q1_patient_id").reset_index(drop=True)
        stored_path = run_v2 / f"{model}_{selection}_top{int(panel)}_patient_predictions.tsv"
        stored = pd.read_csv(stored_path, sep="\t").sort_values("q1_patient_id").reset_index(drop=True)
        same_ids = reproduced["q1_patient_id"].astype(str).equals(stored["q1_patient_id"].astype(str))
        mismatches = len(reproduced) if not same_ids else int(
            ((reproduced[["y_true", "y_pred"]].to_numpy() != stored[["y_true", "y_pred"]].to_numpy()).any(1)).sum()
        )
        stored_mismatch_total += mismatches
        metrics = table_metrics(reproduced["y_true"].to_numpy(int), reproduced["y_pred"].to_numpy(int))
        comparisons.append(
            {
                "model": model,
                "selection": selection,
                "panel_size": int(panel),
                "n_patients": len(reproduced),
                "stored_patient_table_mismatches": mismatches,
                **metrics,
            }
        )

    manifest = json.loads((run_v2 / "run_manifest.json").read_text(encoding="utf-8"))
    input_hash_checks = {
        "samples_locked": sha256(locked_path)
        == manifest["inputs_sha256"]["samples_locked"],
        "split_summary": sha256(workspace / "data" / "tcga_450k" / "split_summary.json")
        == manifest["inputs_sha256"]["split_summary"],
        "E1_patient_predictions": sha256(root / "E1_panel_curve" / "run" / "patient_oof_predictions_by_repeat.tsv.gz")
        == manifest["inputs_sha256"]["E1_patient_predictions"],
    }
    result = {
        "raw_prediction_files": len(raw_frames),
        "raw_rows": len(raw),
        "raw_unmapped_rows": raw_unmapped_rows,
        "raw_true_label_mapping_mismatches": raw_mapping_mismatches,
        "stored_patient_table_mismatches": stored_mismatch_total,
        "input_hash_checks": input_hash_checks,
        "manifest_samples_locked_available": False,
        "independent_mapping_source": (
            "samples_used.tsv physical row -> "
            "tcga_participant::{project_id}::{submitter_id[:12]}"
        ),
        "aggregation": manifest.get("aggregation"),
        "bootstrap_note": "CIs resample patients, but the reported point estimate is one majority vote across all repeats rather than a mean across repeat-specific OOF estimates.",
    }
    return result, pd.DataFrame(comparisons)


def audit_e4(root: Path, workspace: Path, probe_ids: np.ndarray) -> dict:
    e4 = root / "E4_contrastive_shap" / "run_v2"
    pairs = pd.read_csv(e4 / "contrastive_attribution_pairs.tsv", sep="\t")
    details = pd.read_csv(e4 / "pair_probe_fold_details.tsv.gz", sep="\t")

    pair_valid = pairs["probe_index"].between(0, len(probe_ids) - 1)
    pair_mapping = np.zeros(len(pairs), dtype=bool)
    pair_mapping[pair_valid] = (
        probe_ids[pairs.loc[pair_valid, "probe_index"].astype(int).to_numpy()]
        == pairs.loc[pair_valid, "probe_id"].astype(str).to_numpy()
    )
    detail_valid = details["probe_index"].between(0, len(probe_ids) - 1)
    detail_mapping = np.zeros(len(details), dtype=bool)
    detail_mapping[detail_valid] = (
        probe_ids[details.loc[detail_valid, "probe_index"].astype(int).to_numpy()]
        == details.loc[detail_valid, "probe_id"].astype(str).to_numpy()
    )

    selected_dir = root / "E1_panel_curve" / "run" / "selected_features"
    fold_sets: dict[tuple[int, int, int], set[int]] = {}
    for path in selected_dir.glob("selected_top10000_r*f*.tsv"):
        found = re.search(r"r(\d+)f(\d+)", path.stem)
        if not found:
            continue
        selected = pd.read_csv(path, sep="\t").sort_values("rank")
        for panel in (500, 1000):
            fold_sets[(int(found[1]), int(found[2]), panel)] = set(
                selected.head(panel)["probe_index"].astype(int)
            )
    detail_in_panel = np.array(
        [
            int(row.probe_index) in fold_sets.get((int(row.repeat), int(row.fold), int(row.panel_size)), set())
            for row in details.itertuples(index=False)
        ],
        dtype=bool,
    )

    grouped = details.groupby(
        ["panel_size", "class_a", "class_b", "probe_index", "probe_id"], as_index=False
    ).agg(
        detail_selected_fold_count=("fold", "size"),
        detail_patient_repeat_units=("n_patients_in_fold", "sum"),
        weighted_contrast_numerator=(
            "fold_mean_contrast_A_minus_B",
            lambda values: 0.0,
        ),
    )
    # Pandas named aggregation cannot access the fold-specific weights in the
    # same lambda, so compute the weighted mean separately.
    weighted = (
        details.assign(
            weighted=details["fold_mean_contrast_A_minus_B"] * details["n_patients_in_fold"]
        )
        .groupby(["panel_size", "class_a", "class_b", "probe_index", "probe_id"], as_index=False)
        .agg(weighted_sum=("weighted", "sum"), weight=("n_patients_in_fold", "sum"))
    )
    grouped = grouped.drop(columns="weighted_contrast_numerator").merge(weighted, how="left")
    grouped["detail_weighted_contrast"] = grouped["weighted_sum"] / grouped["weight"]
    merged = pairs.merge(
        grouped,
        on=["panel_size", "class_a", "class_b", "probe_index", "probe_id"],
        how="left",
        validate="one_to_one",
    )
    samples_used = pd.read_csv(workspace / "data" / "tcga_450k" / "samples_used.tsv", sep="\t")
    fold_rows = pd.read_csv(workspace / "data" / "tcga_450k" / "fold_assignments.tsv", sep="\t")[
        "matrix_row"
    ].drop_duplicates().astype(int)
    locked = samples_used.iloc[fold_rows].copy()
    locked["q1_patient_id"] = locked["submitter_id"].astype(str).str.slice(0, 12)
    expected_patients = {
        tuple(sorted((a, b))): locked.loc[
            locked["project_id"].isin([a, b]), "q1_patient_id"
        ].astype(str).nunique()
        for a, b in pairs[["class_a", "class_b"]].drop_duplicates().itertuples(index=False, name=None)
    }
    patient_count_mismatches = sum(
        int(row.n_patients) != expected_patients[tuple(sorted((row.class_a, row.class_b)))]
        for row in pairs.itertuples(index=False)
    )
    return {
        "pair_rows": len(pairs),
        "detail_rows": len(details),
        "pair_global_mapping_fraction": float(pair_mapping.mean()),
        "detail_global_mapping_fraction": float(detail_mapping.mean()),
        "detail_selected_panel_membership_fraction": float(detail_in_panel.mean()),
        "selected_fold_count_mismatches": int(
            (merged["selected_fold_count"] != merged["detail_selected_fold_count"]).sum()
        ),
        "patient_repeat_count_mismatches": int(
            (merged["n_patient_repeat_units"] != merged["detail_patient_repeat_units"]).sum()
        ),
        "max_weighted_contrast_difference": float(
            np.max(np.abs(merged["mean_contrast_A_minus_B"] - merged["detail_weighted_contrast"]))
        ),
        "pair_patient_count_mismatches": int(patient_count_mismatches),
    }


def audit_e5(root: Path) -> tuple[dict, pd.DataFrame]:
    e5 = root / "E5_epic_external"
    aggregate = e5 / "aggregate_v2"
    formal = pd.read_csv(aggregate / "patient_level_validation_formal.tsv", sep="\t")
    exploratory = pd.read_csv(aggregate / "patient_level_validation_exploratory.tsv", sep="\t")
    coad = pd.read_csv(aggregate / "coadread_endpoint_recomputed.tsv", sep="\t")
    coverage = pd.read_csv(aggregate / "panel_probe_coverage.tsv", sep="\t")
    preprocessing = json.loads(
        (e5 / "prepared" / "external_preprocessing_manifest.json").read_text(encoding="utf-8")
    )["prepared"]

    prediction_rows: list[dict] = []
    model_hashes: dict[int, set[str]] = {500: set(), 1000: set()}
    for cohort_dir in sorted((e5 / "evaluated").iterdir()):
        run = cohort_dir / "lr_panel_frozen_20260721"
        pred = pd.read_csv(run / "external_predictions.tsv.gz", sep="\t")
        model_manifest = json.loads((run / "external_model_manifest.json").read_text(encoding="utf-8"))
        prob_cols = [c for c in pred if c.startswith(PROB_E5)]
        for panel, frame in pred.groupby("panel_name", sort=True):
            probabilities = frame[prob_cols].to_numpy(float)
            accepted = frame["accepted_labels"].astype(str).str.split(";")
            recomputed = np.array(
                [predicted in labels for predicted, labels in zip(frame["pred_label_open_33_class"], accepted)]
            )
            size = 500 if "500" in panel else 1000
            entry = model_manifest["models"][panel]
            model_hashes[size].add(entry["development_panel_matrix_sha256"])
            prediction_rows.append(
                {
                    "cohort": cohort_dir.name,
                    "panel": panel,
                    "n_samples": len(frame),
                    "max_probability_sum_error": float(np.max(np.abs(probabilities.sum(1) - 1))),
                    "stored_primary_endpoint_mismatches": int(
                        (recomputed != frame["primary_endpoint_correct"].astype(bool).to_numpy()).sum()
                    ),
                    "argmax_lineage_hit_rate": float(recomputed.mean()),
                }
            )

    formal_coad = formal[formal["cohort"] == "GSE148766"][
        ["panel", "n_correct_endpoint", "endpoint_accuracy"]
    ].rename(
        columns={
            "n_correct_endpoint": "formal_n_correct",
            "endpoint_accuracy": "formal_accuracy",
        }
    )
    coad_compare = formal_coad.merge(
        coad[["panel", "n_correct_endpoint", "endpoint_accuracy"]].rename(
            columns={
                "n_correct_endpoint": "recomputed_n_correct",
                "endpoint_accuracy": "recomputed_accuracy",
            }
        ),
        on="panel",
        how="outer",
    )
    preprocessing_unavailable_counts = {
        row["cohort"]: sum(
            isinstance(value, str) and value.startswith("not_available_in_source")
            for value in row.values()
        )
        for row in preprocessing
    }
    sensitivity = pd.read_csv(e5 / "probe_loss_sensitivity" / "probe_loss_sensitivity.tsv", sep="\t")
    duplicate_sensitivity_profiles = (
        sensitivity.groupby(["n_probes_dropped", "panel"])[
            ["patient_accuracy", "e1_full_panel_patient_acc", "delta_acc_vs_full"]
        ]
        .nunique()
        .max(axis=1)
        .eq(1)
        .sum()
    )
    result = {
        "formal_rows": len(formal),
        "exploratory_rows": len(exploratory),
        "formal_cohort_status": sorted(formal["cohort"].unique().tolist()),
        "exploratory_cohort_status": sorted(exploratory["cohort"].unique().tolist()),
        "coadread_formal_table_disagrees_with_recomputed_rows": int(
            (coad_compare["formal_n_correct"] != coad_compare["recomputed_n_correct"]).sum()
        ),
        "preprocessing_not_available_counts": preprocessing_unavailable_counts,
        "model_development_matrix_hash_count": {str(k): len(v) for k, v in model_hashes.items()},
        "minimum_panel_coverage": float(coverage["coverage"].min()),
        "probe_loss_sensitivity_interpretation": (
            "Internal TCGA patient-level masking experiment keyed to each cohort's missing-probe set; "
            "it is not an external-cohort performance estimate."
        ),
        "duplicate_probe_loss_profiles": int(duplicate_sensitivity_profiles),
        "raw_external_matrices_archived": False,
    }
    return result, pd.DataFrame(prediction_rows), coad_compare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--verify-all-hashes", action="store_true")
    args = parser.parse_args()
    root = args.archive_root.resolve()
    workspace = args.workspace.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    probe_ids = pd.read_csv(workspace / "data" / "tcga_450k" / "probe_ids.tsv", sep="\t")[
        "probe_id"
    ].astype(str).to_numpy()
    manifest_audit = verify_manifest(root) if args.verify_all_hashes else {"not_run": True}
    e2, e2_predictions = audit_e2(root, probe_ids)
    e3, e3_comparisons = audit_e3(root, workspace)
    e4 = audit_e4(root, workspace, probe_ids)
    e5, e5_predictions, coad_compare = audit_e5(root)

    e2_predictions.to_csv(out / "e2_prediction_integrity.tsv", sep="\t", index=False)
    e2_predictions.attrs["recovered"].to_csv(
        out / "e2_recovered_method_metrics_by_repeat.tsv", sep="\t", index=False
    )
    e2_predictions.attrs["recovered_summary"].to_csv(
        out / "e2_recovered_method_summary.tsv", sep="\t", index=False
    )
    e2_predictions.attrs["aggregate_compare"].to_csv(
        out / "e2_recovered_vs_packaged_aggregate.tsv", sep="\t", index=False
    )
    e3_comparisons.to_csv(out / "e3_reproduction_comparison.tsv", sep="\t", index=False)
    e5_predictions.to_csv(out / "e5_prediction_integrity.tsv", sep="\t", index=False)
    coad_compare.to_csv(out / "e5_coadread_table_disagreement.tsv", sep="\t", index=False)
    summary = {
        "verification_status": "ANALYZED",
        "archive_sha256": sha256(args.archive.resolve()),
        "archive_manifest": manifest_audit,
        "E2": e2,
        "E3": e3,
        "E4": e4,
        "E5": e5,
    }
    (out / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
