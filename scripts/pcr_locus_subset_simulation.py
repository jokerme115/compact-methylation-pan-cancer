#!/usr/bin/env python3
"""Diagnostic: how usable is a PCR-format (few-locus) cohort for this panel?

Two questions, both answered with TCGA 450k data already locked in this repo:

A. FROZEN-MODEL SUBSTITUTION
   Take the existing LR panel pipeline trained on the full 1048-locus stable
   panel. At test time observe only k loci and leave the rest as NaN, so the
   pipeline's own SimpleImputer(strategy="median") fills them from the training
   fold. This is exactly what happens if a cohort measured on a k-locus PCR
   assay is fed to the published model.

B. COARSE-ENDPOINT RETRAINING
   Collapse the 33 TCGA classes into coarser endpoints and rerun the standard
   fold-internal variance selection + LR. Tests whether a cheaper assay can
   support a less granular clinical question.

This is an INTERNAL DIAGNOSTIC. It is not a manuscript result and must not be
placed in submission tables without a separate decision (see AGENTS.md 第五条).

Substrate: data/tcga_450k/X_tumor_stable_panel.npy (9065 x 1048, tumour only).
Because every locus in this matrix is already panel-grade informative, all
k-locus numbers below are OPTIMISTIC upper bounds for an arbitrary k-locus
assay -- the real PCR case is no better than what is reported here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROMOTER_REGIONS = {"TSS1500", "TSS200", "1stExon", "5'UTR"}

# Illustrative organ-system coarsening of the 33 TCGA project codes.
# Written out explicitly so it can be audited or swapped.
ORGAN9 = {
    "肺与胸膜": ["TCGA-LUAD", "TCGA-LUSC", "TCGA-MESO"],
    "消化道": ["TCGA-ESCA", "TCGA-STAD", "TCGA-COAD", "TCGA-READ",
               "TCGA-LIHC", "TCGA-CHOL", "TCGA-PAAD"],
    "乳腺与妇科": ["TCGA-BRCA", "TCGA-OV", "TCGA-UCEC", "TCGA-UCS", "TCGA-CESC"],
    "泌尿与男性生殖": ["TCGA-BLCA", "TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH",
                       "TCGA-PRAD", "TCGA-TGCT"],
    "头颈与皮肤眼": ["TCGA-HNSC", "TCGA-SKCM", "TCGA-UVM"],
    "中枢神经": ["TCGA-GBM", "TCGA-LGG"],
    "内分泌": ["TCGA-THCA", "TCGA-PCPG", "TCGA-ACC", "TCGA-THYM"],
    "软组织与骨": ["TCGA-SARC"],
    "血液与淋巴": ["TCGA-LAML", "TCGA-DLBC"],
}

# The decision that actually drives first-line therapy in CUP work.
ADENO_SQUAM_OTHER = {
    "腺癌样": ["TCGA-LUAD", "TCGA-COAD", "TCGA-READ", "TCGA-STAD", "TCGA-PAAD",
               "TCGA-CHOL", "TCGA-LIHC", "TCGA-PRAD", "TCGA-UCEC", "TCGA-OV",
               "TCGA-BRCA", "TCGA-THCA", "TCGA-KIRC", "TCGA-KIRP"],
    "鳞癌样": ["TCGA-LUSC", "TCGA-HNSC", "TCGA-ESCA", "TCGA-CESC", "TCGA-BLCA"],
    "其他": ["TCGA-GBM", "TCGA-LGG", "TCGA-MESO", "TCGA-SKCM", "TCGA-UVM",
             "TCGA-SARC", "TCGA-LAML", "TCGA-DLBC", "TCGA-ACC", "TCGA-PCPG",
             "TCGA-THYM", "TCGA-TGCT", "TCGA-UCS", "TCGA-KICH"],
}

SCHEMES = {"organ9": ORGAN9, "adeno_squam_other": ADENO_SQUAM_OTHER}


def build_lr(seed: int, n_jobs: int, max_iter: int) -> Pipeline:
    """Byte-for-byte the pipeline used by src/models/q1_lr_cpg_panel.py."""
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


def top3_accuracy(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    order = np.argsort(-proba, axis=1)[:, :3]
    hit = (classes[order] == y[:, None]).any(axis=1)
    return float(hit.mean())


ISLAND_COL = "Relation_to_UCSC_CpG_Island"


def load_probe_annotation(path: Path, probe_ids: list[str]) -> pd.DataFrame:
    """GPL13534 manifest: 7 preamble lines, then a 33-column table headed IlmnID."""
    df = pd.read_csv(
        path, compression="gzip", skiprows=7, low_memory=False, on_bad_lines="skip",
        usecols=["IlmnID", "UCSC_RefGene_Group", ISLAND_COL],
    )
    df = df.rename(columns={"IlmnID": "probe_id"})
    return df[df["probe_id"].isin(set(probe_ids))].drop_duplicates("probe_id")


def load_label_map(scheme: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, members in scheme.items():
        for code in members:
            mapping[code] = group
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-npy", required=True)
    ap.add_argument("--probe-ids", required=True)
    ap.add_argument("--samples-tsv", required=True,
                    help="Locked tumour-only manifest, row-aligned with --matrix-npy.")
    ap.add_argument("--samples-full-tsv", required=True,
                    help="Full manifest that fold_assignments.tsv matrix_row indexes into.")
    ap.add_argument("--splits-tsv", required=True)
    ap.add_argument("--annotation-csv-gz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-col", default="project_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-fold-runs", type=int, default=0)
    ap.add_argument("--obs-sizes", default="1,3,5,10,20,50")
    ap.add_argument("--random-draws", type=int, default=5)
    ap.add_argument("--coarse-sizes", default="5,10,20,50,100")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_ids = [l.strip() for l in Path(args.probe_ids).read_text().splitlines() if l.strip()]
    probe_ids = [p for p in probe_ids if p != "probe_id"]
    x = np.asarray(np.load(args.matrix_npy))
    samples = pd.read_csv(args.samples_tsv, sep="\t")
    splits = pd.read_csv(args.splits_tsv, sep="\t")

    if x.shape[0] != len(samples):
        raise SystemExit(f"row mismatch: X={x.shape} samples={len(samples)}")
    if x.shape[1] != len(probe_ids):
        raise SystemExit(f"col mismatch: X={x.shape} probes={len(probe_ids)}")

    y_raw = samples[args.label_col].to_numpy()

    # fold_assignments.matrix_row indexes the FULL manifest; the panel matrix is
    # the tumour-only compaction of it (see scripts/extract_tumor_stable_panel_matrix.py).
    full = pd.read_csv(args.samples_full_tsv, sep="\t")
    tumor_mask = full["tissue_type"].astype(str).str.lower().eq("tumor").to_numpy()
    tumor_rows_full = np.flatnonzero(tumor_mask)
    if len(tumor_rows_full) != x.shape[0]:
        raise SystemExit(f"tumour rows in full manifest ({len(tumor_rows_full)}) "
                         f"!= matrix rows ({x.shape[0]})")
    full_to_locked = np.full(len(full) + 1, -1, dtype=np.int64)
    full_to_locked[tumor_rows_full] = np.arange(len(tumor_rows_full))
    splits["locked_row"] = full_to_locked[splits["matrix_row"].to_numpy()]
    if (splits["locked_row"] < 0).any():
        raise SystemExit("some fold rows are not tumour rows in the full manifest")
    ann = load_probe_annotation(Path(args.annotation_csv_gz), probe_ids)
    ann = ann.set_index("probe_id").reindex(probe_ids).reset_index()
    n_obs = int(ann[ISLAND_COL].notna().sum())
    is_promoter_island = np.zeros(len(probe_ids), dtype=bool)
    island = ann[ISLAND_COL].fillna("").str.contains("Island", regex=False)
    grp = ann["UCSC_RefGene_Group"].fillna("")
    region = grp.apply(lambda s: any(r in s for r in PROMOTER_REGIONS))
    is_promoter_island = (island & region).to_numpy()

    print(f"[info] X={x.shape} probes_with_annotation={n_obs} "
          f"promoter_island={int(is_promoter_island.sum())}", flush=True)

    obs_sizes = [int(v) for v in args.obs_sizes.split(",")]
    coarse_sizes = [int(v) for v in args.coarse_sizes.split(",")]
    classes_full = np.unique(y_raw)
    rng = np.random.default_rng(args.seed)

    frozen_rows: list[dict] = []
    coarse_rows: list[dict] = []

    fold_keys = (splits[["repeat", "fold"]].drop_duplicates().sort_values(["repeat", "fold"]))
    if args.max_fold_runs:
        fold_keys = fold_keys.head(args.max_fold_runs)

    for n_run, (repeat, fold) in enumerate(fold_keys.itertuples(index=False), start=1):
        if repeat >= args.repeats:
            continue
        sub = splits[(splits["repeat"] == repeat) & (splits["fold"] == fold)]
        tr = sub[sub["split"] == "train"]["locked_row"].to_numpy()
        te = sub[sub["split"] == "test"]["locked_row"].to_numpy()
        x_tr, x_te = x[tr], x[te]
        y_tr, y_te = y_raw[tr], y_raw[te]
        seed = args.seed + int(repeat) * 100 + int(fold)

        # ---------- A. frozen model, partial observation ----------
        pipe = build_lr(seed, args.n_jobs, args.max_iter).fit(x_tr, y_tr)
        coef = pipe.named_steps["model"].coef_
        importance = np.abs(coef).sum(axis=0)
        order_all = np.argsort(-importance)
        order_prom = np.argsort(-np.where(is_promoter_island, importance, -np.inf))
        frozen_classes = pipe.named_steps["model"].classes_

        def evaluate(obs_idx: np.ndarray) -> dict:
            keep = np.zeros(len(probe_ids), dtype=bool)
            keep[obs_idx] = True
            z = x_te.copy()
            z[:, ~keep] = np.nan
            proba = pipe.predict_proba(z)
            pred = frozen_classes[np.argmax(proba, axis=1)]
            return {
                "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_te, pred)),
                "top3_accuracy": top3_accuracy(proba, y_te, frozen_classes),
            }

        for k in obs_sizes:
            k = min(k, len(probe_ids))
            for mode, idx in (
                ("oracle_topk", order_all[:k]),
                ("promoter_topk", order_prom[:k]),
            ):
                row = {"repeat": int(repeat), "fold": int(fold), "k": k, "mode": mode}
                row.update(evaluate(idx))
                frozen_rows.append(row)
            # Apples-to-apples control: same k loci, but the model is retrained on
            # them instead of being asked to survive median imputation.
            oracle_idx = order_all[:k]
            p_retrain = build_lr(seed, args.n_jobs, args.max_iter).fit(
                x_tr[:, oracle_idx], y_tr)
            pred_retrain = p_retrain.predict(x_te[:, oracle_idx])
            frozen_rows.append({
                "repeat": int(repeat), "fold": int(fold), "k": k,
                "mode": "oracle_topk_retrained",
                "macro_f1": float(f1_score(y_te, pred_retrain, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_te, pred_retrain)),
                "top3_accuracy": top3_accuracy(
                    p_retrain.predict_proba(x_te[:, oracle_idx]),
                    y_te, p_retrain.named_steps["model"].classes_),
            })
            for draw in range(args.random_draws):
                idx = rng.choice(len(probe_ids), size=k, replace=False)
                row = {"repeat": int(repeat), "fold": int(fold), "k": k,
                       "mode": f"random_draw{draw}"}
                row.update(evaluate(idx))
                frozen_rows.append(row)
        # full panel reference
        row = {"repeat": int(repeat), "fold": int(fold), "k": len(probe_ids), "mode": "all_loci"}
        row.update(evaluate(np.arange(len(probe_ids))))
        frozen_rows.append(row)

        # ---------- B. coarse endpoint, retrained ----------
        for scheme_name, scheme in SCHEMES.items():
            lmap = load_label_map(scheme)
            y_tr_g = np.array([lmap.get(v, "其他") for v in y_tr])
            y_te_g = np.array([lmap.get(v, "其他") for v in y_te])
            var = x_tr.var(axis=0)
            order_var = np.argsort(-var)
            for k in coarse_sizes + [len(probe_ids)]:
                k = min(k, len(probe_ids))
                sel = order_var[:k]
                p = build_lr(seed, args.n_jobs, args.max_iter).fit(x_tr[:, sel], y_tr_g)
                pred = p.predict(x_te[:, sel])
                coarse_rows.append({
                    "repeat": int(repeat), "fold": int(fold), "scheme": scheme_name,
                    "n_groups": len(set(y_tr_g.tolist())), "k": k,
                    "macro_f1": float(f1_score(y_te_g, pred, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(y_te_g, pred)),
                })
            if scheme_name == "organ9":
                p = build_lr(seed, args.n_jobs, args.max_iter).fit(x_tr, y_tr)
                pred = p.predict(x_te)
                coarse_rows.append({
                    "repeat": int(repeat), "fold": int(fold), "scheme": "full_33class",
                    "n_groups": len(set(y_tr.tolist())), "k": len(probe_ids),
                    "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
                    "accuracy": float(accuracy_score(y_te, pred)),
                })
        print(f"[run {n_run}/{len(fold_keys)}] repeat={repeat} fold={fold} done", flush=True)

    frozen = pd.DataFrame(frozen_rows)
    coarse = pd.DataFrame(coarse_rows)

    def agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        g = df.groupby(keys, as_index=False).agg(
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
            **({"top3_accuracy_mean": ("top3_accuracy", "mean")} if "top3_accuracy" in df else {}),
        )
        return g.round(4)

    frozen_summary = agg(frozen.drop(columns=["repeat", "fold"]).assign(
        mode=lambda d: np.where(d["mode"].str.startswith("random"), "random", d["mode"])),
        ["mode", "k"])
    coarse_summary = agg(coarse.drop(columns=["repeat", "fold"]), ["scheme", "n_groups", "k"])

    frozen.to_csv(out_dir / "frozen_partial_observation_folds.tsv", sep="\t", index=False)
    frozen_summary.to_csv(out_dir / "frozen_partial_observation_summary.tsv", sep="\t", index=False)
    coarse.to_csv(out_dir / "coarse_endpoint_folds.tsv", sep="\t", index=False)
    coarse_summary.to_csv(out_dir / "coarse_endpoint_summary.tsv", sep="\t", index=False)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "matrix_npy": str(args.matrix_npy),
        "n_samples": int(x.shape[0]), "n_loci": int(x.shape[1]),
        "n_loci_promoter_island": int(is_promoter_island.sum()),
        "label_col": args.label_col,
        "repeats_used": args.repeats, "folds_evaluated": int(len(fold_keys)),
        "obs_sizes": obs_sizes, "coarse_sizes": coarse_sizes,
        "random_draws": args.random_draws, "seed": args.seed,
        "organ9": ORGAN9, "adeno_squam_other": ADENO_SQUAM_OTHER,
        "note": "internal diagnostic; k-locus numbers are optimistic upper bounds",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== frozen model, partial observation ===")
    print(frozen_summary.to_string(index=False))
    print("\n=== coarse endpoint ===")
    print(coarse_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
