from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)


PROB_PREFIX = "probability_class_"
LABELS = [
    "TCGA-ACC", "TCGA-BLCA", "TCGA-BRCA", "TCGA-CESC", "TCGA-CHOL",
    "TCGA-COAD", "TCGA-DLBC", "TCGA-ESCA", "TCGA-GBM", "TCGA-HNSC",
    "TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP", "TCGA-LAML", "TCGA-LGG",
    "TCGA-LIHC", "TCGA-LUAD", "TCGA-LUSC", "TCGA-MESO", "TCGA-OV",
    "TCGA-PAAD", "TCGA-PCPG", "TCGA-PRAD", "TCGA-READ", "TCGA-SARC",
    "TCGA-SKCM", "TCGA-STAD", "TCGA-TGCT", "TCGA-THCA", "TCGA-THYM",
    "TCGA-UCEC", "TCGA-UCS", "TCGA-UVM",
]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def brier_multiclass(y: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(p.shape[1], dtype=float)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = p.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "brier_multiclass": brier_multiclass(y, p),
        "nll": float(log_loss(y, np.clip(p, 1e-12, 1), labels=np.arange(p.shape[1]))),
    }


def audit_patient_predictions(path: Path, experiment: str, method: str) -> tuple[list[dict], pd.DataFrame]:
    df = pd.read_csv(path, sep="\t")
    prob_cols = sorted(
        [c for c in df if c.startswith(PROB_PREFIX)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    rows: list[dict] = []
    for (panel, repeat), group in df.groupby(["panel_size", "repeat"], sort=True):
        p = group[prob_cols].to_numpy(float)
        y = group["y_true"].to_numpy(int)
        recomputed = metrics(y, p)
        rows.append({
            "experiment": experiment,
            "method": method,
            "panel_size": int(panel),
            "repeat": int(repeat),
            "n_rows": len(group),
            "n_patients": group["q1_patient_id"].nunique(),
            "duplicate_patient_rows": int(group.duplicated("q1_patient_id").sum()),
            "max_probability_sum_error": float(np.max(np.abs(p.sum(axis=1) - 1))),
            "stored_pred_mismatch": int(np.sum(p.argmax(axis=1) != group["y_pred"].to_numpy(int))),
            **recomputed,
        })
    return rows, df


def paired_patient_bootstrap(
    df: pd.DataFrame,
    panel_a: int,
    panel_b: int,
    n_boot: int = 2000,
    seed: int = 20260818,
) -> dict[str, float]:
    a = df[df.panel_size == panel_a].copy()
    b = df[df.panel_size == panel_b].copy()
    prob_cols = sorted(
        [c for c in df if c.startswith(PROB_PREFIX)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    key = ["repeat", "q1_patient_id"]
    a = a.set_index(key).sort_index()
    b = b.set_index(key).sort_index()
    if not a.index.equals(b.index):
        raise ValueError(f"Panel {panel_a} and {panel_b} patient-repeat indices differ")
    patient_ids = np.array(sorted(set(a.index.get_level_values("q1_patient_id"))))

    patient_to_pos = {pid: i for i, pid in enumerate(patient_ids)}
    repeat_arrays = []
    for repeat in sorted(set(a.index.get_level_values("repeat"))):
        aa = a.xs(repeat, level="repeat").loc[patient_ids]
        bb = b.xs(repeat, level="repeat").loc[patient_ids]
        y = aa.y_true.to_numpy(int)
        if not np.array_equal(y, bb.y_true.to_numpy(int)):
            raise ValueError("True labels differ between paired panels")
        repeat_arrays.append((y, aa[prob_cols].to_numpy().argmax(1), bb[prob_cols].to_numpy().argmax(1)))

    def delta(weights: np.ndarray) -> float:
        sa, sb = [], []
        for y, pred_a, pred_b in repeat_arrays:
            sa.append(f1_score(y, pred_a, average="macro", zero_division=0, sample_weight=weights))
            sb.append(f1_score(y, pred_b, average="macro", zero_division=0, sample_weight=weights))
        return float(np.mean(sa) - np.mean(sb))

    observed = delta(np.ones(len(patient_ids), dtype=int))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    # Resample patient identities once per bootstrap draw, retaining all repeat predictions.
    for i in range(n_boot):
        sampled_positions = rng.integers(0, len(patient_ids), size=len(patient_ids))
        weights = np.bincount(sampled_positions, minlength=len(patient_ids))
        deltas[i] = delta(weights)
    return {
        "panel_a": panel_a,
        "panel_b": panel_b,
        "metric": "macro_f1",
        "estimate_a_minus_b": observed,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_replicates": n_boot,
    }


RUN_RE = re.compile(r"(?P<model>hierarchical|xgboost)__(?P<method>[^_]+)__(?:top)(?P<topk>\d+)__r(?P<repeat>\d+)f(?P<fold>\d+)")


def audit_and_recover_e3(root: Path, samples_path: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(samples_path, sep="\t")
    samples["patient_id"] = samples["submitter_id"].astype(str).str.slice(0, 12)
    fold_rows: list[dict] = []
    patient_records: list[pd.DataFrame] = []
    for model_dir in [root / "run" / "hierarchical", root / "run" / "xgboost_baseline"]:
        for tsv in sorted((model_dir / "predictions").glob("*.tsv")):
            match = RUN_RE.fullmatch(tsv.stem)
            if not match:
                raise ValueError(f"Unexpected E3 filename: {tsv.name}")
            meta = match.groupdict()
            npz_path = tsv.with_name(tsv.stem + "_proba.npz")
            frame = pd.read_csv(tsv, sep="\t")
            archive = np.load(npz_path)
            keys = list(archive.keys())
            if "y_proba" not in keys:
                raise ValueError(f"Missing y_proba in {npz_path}; got {keys}")
            p = np.asarray(archive["y_proba"], dtype=float)
            if p.shape != (len(frame), len(LABELS)):
                raise ValueError(f"Probability shape mismatch for {tsv.name}: {p.shape}")
            rows = frame.matrix_row.to_numpy(int)
            mapped = samples.iloc[rows]
            true_labels = mapped.project_id.astype(str).to_numpy()
            pred_labels = np.array(LABELS, dtype=object)[p.argmax(1)]
            fold_rows.append({
                **meta,
                "n_rows": len(frame),
                "probability_key": "y_proba",
                "max_probability_sum_error": float(np.max(np.abs(p.sum(1) - 1))),
                "true_label_mapping_mismatch": int(np.sum(true_labels != frame.true_label.astype(str).to_numpy())),
                "pred_label_mapping_mismatch": int(np.sum(pred_labels != frame.pred_label.astype(str).to_numpy())),
                "npz_true_index_mismatch": int(np.sum(archive["y_true"].astype(int) != np.array([LABEL_TO_INT[x] for x in true_labels]))),
                "npz_pred_index_mismatch": int(np.sum(archive["y_pred"].astype(int) != p.argmax(1))),
            })
            temp = pd.DataFrame(p, columns=[f"p_{label}" for label in LABELS])
            temp["patient_id"] = mapped.patient_id.to_numpy()
            temp["true_label"] = true_labels
            temp["model"] = meta["model"]
            temp["method"] = meta["method"]
            temp["topk"] = int(meta["topk"])
            temp["repeat"] = int(meta["repeat"])
            patient_records.append(temp)

    all_sample = pd.concat(patient_records, ignore_index=True)
    pcols = [f"p_{label}" for label in LABELS]
    patient = all_sample.groupby(
        ["model", "method", "topk", "repeat", "patient_id", "true_label"], as_index=False
    )[pcols].mean()
    patient["y_true"] = patient.true_label.map(LABEL_TO_INT)
    patient["y_pred"] = patient[pcols].to_numpy().argmax(1)
    patient.to_csv(out_dir / "e3_recovered_patient_predictions.tsv.gz", sep="\t", index=False, compression="gzip")

    metric_rows: list[dict] = []
    for keys, g in patient.groupby(["model", "method", "topk", "repeat"], sort=True):
        model, method, topk, repeat = keys
        vals = metrics(g.y_true.to_numpy(int), g[pcols].to_numpy(float))
        metric_rows.append({
            "model": model, "method": method, "topk": int(topk), "repeat": int(repeat),
            "n_patients": g.patient_id.nunique(), **vals,
        })
    by_repeat = pd.DataFrame(metric_rows)
    by_repeat.to_csv(out_dir / "e3_recovered_patient_metrics_by_repeat.tsv", sep="\t", index=False)
    summary = by_repeat.groupby(["model", "method", "topk"], as_index=False).agg(
        n_patients=("n_patients", "min"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        accuracy_mean=("accuracy", "mean"), brier_mean=("brier_multiclass", "mean"), nll_mean=("nll", "mean"),
    )
    summary.to_csv(out_dir / "e3_recovered_patient_summary.tsv", sep="\t", index=False)
    return pd.DataFrame(fold_rows), summary


def audit_e4(root: Path, probe_ids_path: Path, e1_root: Path) -> dict:
    pair = pd.read_csv(root / "run" / "contrastive_attribution_pairs.tsv", sep="\t")
    global_probe_ids = pd.read_csv(probe_ids_path, sep="\t").probe_id.astype(str).to_numpy()
    valid_index = pair.probe_index.between(0, len(global_probe_ids) - 1)
    exact_global_position = np.zeros(len(pair), dtype=bool)
    exact_global_position[valid_index] = (
        global_probe_ids[pair.loc[valid_index, "probe_index"].to_numpy(int)]
        == pair.loc[valid_index, "probe_id"].astype(str).to_numpy()
    )
    selected_union: dict[int, set[str]] = {500: set(), 1000: set()}
    selected_files = sorted((e1_root / "selected_features").glob("selected_top10000_*.tsv"))
    for selected in selected_files:
        selected_table = pd.read_csv(selected, sep="\t").sort_values("rank")
        for panel in selected_union:
            selected_union[panel].update(selected_table.probe_id.astype(str).head(panel).tolist())
    union_membership = [str(pid) in selected_union.get(int(panel), set()) for pid, panel in zip(pair.probe_id, pair.panel_size)]
    return {
        "n_rows": len(pair),
        "fraction_probe_id_equals_global_probe_at_reported_index": float(np.mean(exact_global_position)),
        "fraction_reported_probe_present_in_any_corresponding_fold_panel": float(np.mean(union_membership)),
        "reported_pair_counts": sorted(pair.n_patients_in_pair.unique().astype(int).tolist()),
        "interpretation": "If the first fraction is 1 while panel-union membership is near 0, local panel positions were mapped as global CpG indices.",
    }


def audit_e5(root: Path) -> tuple[pd.DataFrame, dict]:
    aggregate = pd.read_csv(root / "e5_epic_aggregated.tsv", sep="\t")
    rows = []
    model_hashes: dict[str, set[str]] = {"500": set(), "1000": set()}
    for cohort_dir in sorted((root / "evaluated").iterdir()):
        run = cohort_dir / "lr_panel_frozen_20260721"
        pred = pd.read_csv(run / "external_predictions.tsv.gz", sep="\t")
        pre = json.loads((run / "external_preprocessing_manifest.json").read_text(encoding="utf-8"))
        model = json.loads((run / "external_model_manifest.json").read_text(encoding="utf-8"))
        prob_cols = [c for c in pred if c.startswith("probability__")]
        for panel, g in pred.groupby("panel_size"):
            accepted = g.accepted_labels.str.split(";")
            recomputed = np.array([p in labels for p, labels in zip(g.pred_label_open_33_class, accepted)])
            rows.append({
                "cohort": cohort_dir.name,
                "panel_size": int(panel),
                "n_samples": len(g),
                "probability_sum_max_error": float(np.max(np.abs(g[prob_cols].sum(1) - 1))),
                "stored_endpoint_mismatch": int(np.sum(recomputed != g.primary_endpoint_correct.astype(bool).to_numpy())),
                "accuracy_recomputed": float(np.mean(recomputed)),
                "preprocessing_fields_not_recorded": int(sum(v == "not_recorded" for v in pre.values())),
                "formal_or_diagnostic": pre.get("formal_or_diagnostic"),
            })
            entry = model["models"][f"consensus_locked_{int(panel)}_cpg"]
            model_hashes[str(int(panel))].add(entry["development_panel_matrix_sha256"])
    return pd.DataFrame(rows), {k: sorted(v) for k, v in model_hashes.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--e1-only", action="store_true")
    args = parser.parse_args()
    root = args.archive_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    patient_audits: list[dict] = []
    e1_rows, e1_df = audit_patient_predictions(
        root / "E1_panel_curve" / "run" / "patient_oof_predictions_by_repeat.tsv.gz", "E1", "variance"
    )
    patient_audits.extend(e1_rows)
    e2_root = root / "E2_feature_selection" / "run"
    for method_dir in sorted(p for p in e2_root.iterdir() if p.is_dir()):
        pred_path = method_dir / "patient_oof_predictions_by_repeat.tsv.gz"
        if pred_path.exists():
            rows, _ = audit_patient_predictions(pred_path, "E2", method_dir.name)
            patient_audits.extend(rows)
    pd.DataFrame(patient_audits).to_csv(out / "patient_prediction_integrity.tsv", sep="\t", index=False)

    comparisons = []
    for panel in [600, 700, 800, 900, 1500, 2000, 5000, 10000]:
        comparisons.append(paired_patient_bootstrap(e1_df, panel, 1000, args.bootstrap))
    pd.DataFrame(comparisons).to_csv(out / "e1_paired_macro_f1_vs_1000.tsv", sep="\t", index=False)

    if args.e1_only:
        print(json.dumps({"e1_patient_integrity_rows": len(e1_rows), "paired_comparisons": comparisons}, indent=2))
        return

    e3_integrity, e3_summary = audit_and_recover_e3(
        # E3 matrix_row is the physical row in samples_used.tsv (0..9811), not the
        # compact position in the 9,065-row locked subset.
        root / "E3_classifier_benchmark", args.workspace / "data" / "tcga_450k" / "samples_used.tsv", out
    )
    e3_integrity.to_csv(out / "e3_fold_prediction_integrity.tsv", sep="\t", index=False)

    e4 = audit_e4(
        root / "E4_contrastive_shap", args.workspace / "data" / "tcga_450k" / "probe_ids.tsv",
        root / "E1_panel_curve" / "run",
    )
    (out / "e4_mapping_audit.json").write_text(json.dumps(e4, indent=2), encoding="utf-8")
    e5, model_hashes = audit_e5(root / "E5_epic_external")
    e5.to_csv(out / "e5_prediction_integrity.tsv", sep="\t", index=False)

    summary = {
        "archive_root": str(root),
        "archive_sha256": sha256(
            args.workspace
            / "artifacts"
            / "archives"
            / "revision_rerun_deliveries"
            / "revision_rerun_20260815_results.tar.gz"
        ),
        "patient_integrity_max_probability_sum_error": float(pd.DataFrame(patient_audits).max_probability_sum_error.max()),
        "patient_integrity_total_duplicate_patient_rows": int(pd.DataFrame(patient_audits).duplicate_patient_rows.sum()),
        "e3_runs": len(e3_integrity),
        "e3_mapping_mismatches": int(e3_integrity.true_label_mapping_mismatch.sum() + e3_integrity.pred_label_mapping_mismatch.sum()),
        "e3_best_patient_macro_f1": e3_summary.sort_values("macro_f1_mean", ascending=False).head(10).to_dict("records"),
        "e4": e4,
        "e5_model_development_matrix_hashes_by_panel": model_hashes,
        "e5_preprocessing_fields_not_recorded_per_result": sorted(e5.preprocessing_fields_not_recorded.unique().astype(int).tolist()),
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
