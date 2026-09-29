from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def infer_patient_id(df: pd.DataFrame) -> pd.Series:
    """Infer a strict patient key; never fall back to an aliquot or row ID.

    GDC case UUIDs are preferred.  When they are unavailable, the complete
    TCGA participant submitter ID is retained and namespaced by project.  A
    malformed or missing identity is a hard error because a sample-level
    fallback would invalidate patient-level validation.
    """
    project = df.get("project_id", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    uuid_column = next(
        (name for name in ("gdc_case_uuid", "case_uuid", "case_id") if name in df.columns),
        None,
    )
    if uuid_column is not None:
        values = df[uuid_column].fillna("").astype(str).str.strip()
        if values.ne("").all():
            return (project + "::case:" + values).astype(str)

    if "submitter_id" not in df.columns:
        raise ValueError(
            "Patient identity is unavailable: expected GDC case UUID or complete submitter_id; "
            "sample/file IDs and row numbers are not valid patient fallbacks."
        )
    values = df["submitter_id"].fillna("").astype(str).str.strip()
    valid = values.str.fullmatch(r"TCGA-[^-\s]+-[^-\s]+")
    if not bool(valid.all()):
        bad = values[~valid].head(5).tolist()
        raise ValueError(f"Malformed or missing TCGA participant submitter_id values: {bad}")
    return (project + "::participant:" + values).astype(str)


def build_patient_identity_audit(df: pd.DataFrame, folds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return auditable identity and repeated-measurement checks."""
    patient_counts = df.groupby("q1_patient_id", dropna=False).size()
    participant = df["q1_participant_id"]
    participant_project_counts = df.groupby("q1_participant_id")["project_id"].nunique()
    rows: list[dict[str, object]] = [
        {"metric": "n_samples", "value": int(len(df)), "details": "filtered development cohort"},
        {"metric": "n_patients", "value": int(patient_counts.size), "details": "strict patient key"},
        {"metric": "max_samples_per_patient", "value": int(patient_counts.max()), "details": "repeated samples audit"},
        {"metric": "n_patients_with_multiple_samples", "value": int((patient_counts > 1).sum()), "details": "repeated samples audit"},
        {"metric": "n_participant_ids_with_multiple_projects", "value": int((participant_project_counts > 1).sum()), "details": "cross-project label conflict"},
        {"metric": "n_duplicate_file_id", "value": int(df["file_id"].duplicated().sum()) if "file_id" in df.columns else 0, "details": "duplicate aliquot proxy"},
        {"metric": "n_duplicate_file_name", "value": int(df["file_name"].duplicated().sum()) if "file_name" in df.columns else 0, "details": "repeated measurement proxy"},
        {"metric": "n_duplicate_submitter_id", "value": int(df["submitter_id"].duplicated().sum()) if "submitter_id" in df.columns else 0, "details": "participant barcode duplicate"},
    ]
    if folds is not None and not folds.empty:
        fold_patient = folds.merge(df[["matrix_row", "q1_patient_id"]], on="matrix_row", how="left")
        cross_fold = (
            fold_patient.groupby(["repeat", "q1_patient_id"])["fold"].nunique().gt(1).sum()
        )
        rows.append({"metric": "n_patient_repeat_assignments_spanning_folds", "value": int(cross_fold), "details": "must be zero"})
    return pd.DataFrame(rows)


def filtered_samples(samples_tsv: Path, label_col: str, tissue_type: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(samples_tsv, sep="\t")
    if tissue_type:
        df = df[df["tissue_type"].fillna("").str.lower() == tissue_type.lower()].copy()
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    df = df.reset_index().rename(columns={"index": "matrix_row"})
    labels = LabelEncoder()
    y = labels.fit_transform(df[label_col].astype(str).to_numpy())
    df["q1_label_id"] = y
    df["q1_patient_id"] = infer_patient_id(df)
    if "submitter_id" in df.columns:
        df["q1_participant_id"] = df["submitter_id"].fillna("").astype(str).str.strip()
    else:
        df["q1_participant_id"] = df["q1_patient_id"].astype(str)
    participant_project_counts = df.groupby("q1_participant_id")[label_col].nunique()
    conflicts = participant_project_counts[participant_project_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Participant IDs map to multiple cancer labels/projects; resolve identity conflicts before splitting: "
            f"{conflicts.index.tolist()[:10]}"
        )
    return df, y, labels.classes_.astype(str).tolist()


def make_repeated_group_folds(
    df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    groups = df["q1_patient_id"].astype(str).to_numpy()
    use_groups = len(np.unique(groups)) < len(groups)
    for repeat in range(repeats):
        splitter_seed = seed + repeat
        if use_groups:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=splitter_seed)
            split_iter = splitter.split(np.zeros(len(y)), y, groups=groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=splitter_seed)
            split_iter = splitter.split(np.zeros(len(y)), y)
        for fold, (train_idx, test_idx) in enumerate(split_iter):
            for idx in train_idx:
                rows.append({"repeat": repeat, "fold": fold, "matrix_row": int(df.loc[idx, "matrix_row"]), "split": "train"})
            for idx in test_idx:
                rows.append({"repeat": repeat, "fold": fold, "matrix_row": int(df.loc[idx, "matrix_row"]), "split": "test"})
    return pd.DataFrame(rows)


def write_split_artifacts(
    samples_tsv: Path,
    out_dir: Path,
    label_col: str = "project_id",
    tissue_type: str = "tumor",
    n_splits: int = 5,
    repeats: int = 3,
    seed: int = 42,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df, y, labels = filtered_samples(samples_tsv, label_col=label_col, tissue_type=tissue_type)
    folds = make_repeated_group_folds(df, y, n_splits=n_splits, repeats=repeats, seed=seed)
    audit = build_patient_identity_audit(df, folds)
    bad_cross_fold = audit.loc[audit["metric"] == "n_patient_repeat_assignments_spanning_folds", "value"]
    if not bad_cross_fold.empty and int(bad_cross_fold.iloc[0]) != 0:
        raise ValueError("Patient identity audit failed: at least one patient spans multiple folds in a repeat")
    df.to_csv(out_dir / "samples_locked.tsv", sep="\t", index=False)
    folds.to_csv(out_dir / "fold_assignments.tsv", sep="\t", index=False)
    audit.to_csv(out_dir / "patient_identity_audit.tsv", sep="\t", index=False)
    summary = {
        "samples_tsv": str(samples_tsv.resolve()),
        "label_col": label_col,
        "tissue_type": tissue_type,
        "n_samples": int(len(df)),
        "n_classes": int(len(labels)),
        "labels": labels,
        "n_splits": int(n_splits),
        "repeats": int(repeats),
        "seed": int(seed),
        "patient_id_column": "q1_patient_id",
        "class_counts": {str(k): int(v) for k, v in df[label_col].value_counts().sort_index().to_dict().items()},
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
