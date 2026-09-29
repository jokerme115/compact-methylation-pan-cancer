"""E6 pilot: redundancy-aware panel selection inside the outer-training fold.

Motivation
----------
The frozen pipeline ranks CpGs by a *marginal* score (variance / ANOVA /
mutual information / one-vs-rest effect size) and keeps the top-k.  CpGs on the
same CpG island are strongly co-methylated, so a marginal ranking admits large
blocks of near-duplicate probes: the panel spends its budget on one underlying
signal and its cross-fold composition is unstable.

This pilot asks one question before any full-scale design work:

    Holding the candidate ranking fixed, does adding a non-redundancy
    constraint change (i) macro-F1, (ii) cross-fold composition stability and
    (iii) mean intra-panel absolute correlation?

Arms
----
- ``variance_full``  variance top-k over the full candidate space (matches the
  E1 / frozen-panel contract).
- ``dedup``          walk the same fold-internal variance ranking but accept a
  probe only when its maximum absolute correlation against the probes already
  accepted is <= ``tau``.  The walk starts from a variance top-``prefilter``
  pool so that the pairwise correlation block stays computable.

The comparison is clean: both arms rank by exactly the same statistic, and
``dedup`` only adds a non-redundancy constraint.  Truncating that same ranking
at ``prefilter`` is deliberately *not* an arm -- it reproduces
``variance_full`` exactly, because the pool is itself the head of the ranking
it would be truncated from.

Leakage contract (identical to E1/E2)
-------------------------------------
Every scoring, pooling, de-duplication and model fit happens on the
outer-training partition only.  Candidate rankings are rebuilt from scratch in
each fold.  No external cohort participates in selection.

Runtime design
--------------
A microbenchmark on this dataset (7253 x 413341 float32) showed the naive
implementation is dominated by two artefacts rather than by modelling work:

- ``x[rows]`` with unsorted row indices gathers 11.6 GB at 49 MB/s (234 s),
  because the fold row order is the shuffled output of StratifiedGroupKFold.
  Sorting the row indices turns the gather into a sequential copy.
- ``np.nanvar(..., axis=0)`` costs 128.7 s against 10.6 s for ``np.var`` on the
  same access pattern, and 196 s even after transposing to contiguous rows: the
  mask construction dominates.  ``fast_nanvar_axis0`` reproduces it with a
  blocked sum/sumsq formulation.

Arms therefore share a single matrix gather and a single candidate ranking per
fold, instead of re-reading and re-ranking per arm.

Outputs
-------
``<out-dir>/<arm>/`` mirrors the E2 layout (fold metrics, OOF predictions,
selected-probe manifest, selection stability, repeated-OOF evaluation) plus:

- ``panel_redundancy.tsv``  per fold/panel mean intra-panel |r| and the number
  of probes that survived the redundancy filter before any fill-in.
- ``selection_stability.tsv`` mean pairwise Jaccard across fold panels.

``<out-dir>/parity_summary.tsv`` collects the redundancy and stability
diagnostics per arm and panel size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/data", "src/features", "src/models", "src/evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from q1_feature_selection import select_features  # noqa: E402
from q1_evaluate_repeated_oof import evaluate as evaluate_repeated_oof  # noqa: E402
from q1_lr_cpg_panel import model_predict_proba as full_grid_proba  # noqa: E402
from q1_metrics import calibration_metrics, multiclass_metrics, uncertainty_frame  # noqa: E402
from q1_train_ml_baselines import fold_pairs, labels_for_rows, subset_rows  # noqa: E402

ARMS = ("variance_full", "dedup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E6 redundancy-aware selection pilot.")
    parser.add_argument("--matrix-npy", required=True)
    parser.add_argument("--probe-ids", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arms", default="all", help="Comma-separated arms or 'all'.")
    parser.add_argument(
        "--score-impl",
        default="nanvar",
        choices=("fast", "nanvar"),
        help=(
            "nanvar: np.nanvar, bit-identical to the E1/E2 ranking (default). "
            "fast: blocked sum/sumsq, 3.1x faster and agreeing to rtol=1e-5 with an "
            "identical top-k set, but a slightly different intra-ranking order."
        ),
    )
    parser.add_argument(
        "--prefilter",
        type=int,
        default=10000,
        help="Fold-internal candidate pool size the dedup walk starts from.",
    )
    parser.add_argument("--panel-sizes", default="200,500,1000")
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.9,
        help="A probe is rejected when |r| against any already-selected probe exceeds tau.",
    )
    parser.add_argument("--corr-block", type=int, default=2000, help="Correlation matrix block size.")
    parser.add_argument("--var-block", type=int, default=4096, help="Column block size for fast variance.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--max-fold-runs",
        type=int,
        default=0,
        help="Limit the number of outer folds (smoke testing only). 0 means all folds.",
    )
    parser.add_argument(
        "--no-in-memory",
        action="store_true",
        help="Keep the feature matrix memmapped instead of loading it into RAM.",
    )
    parser.add_argument(
        "--no-sort-rows",
        action="store_true",
        help="Keep the shuffled fold row order (much slower gather; for A/B timing only).",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def fast_nanvar_axis0(x: np.ndarray, block: int = 4096) -> np.ndarray:
    """NaN-aware column variance (ddof=0), computed as blocked sum / sum-of-squares.

    ``np.nanvar`` on this matrix costs ~12x ``np.var`` because it materialises a
    full missing-value mask, and stays expensive even when the access pattern is
    made contiguous.  Summing in float64 over float32 blocks keeps the result
    numerically equivalent to ``np.nanvar`` while restoring the ``np.var``-class
    throughput.
    """

    n_samples, n_features = x.shape
    out = np.empty(n_features, dtype=np.float64)
    for start in range(0, n_features, block):
        stop = min(start + block, n_features)
        chunk = np.ascontiguousarray(x[:, start:stop].T)
        mask = np.isnan(chunk)
        counts = np.float64(n_samples) - mask.sum(axis=1, dtype=np.float64)
        filled = np.where(mask, np.float32(0.0), chunk)
        total = filled.sum(axis=1, dtype=np.float64)
        sumsq = np.einsum("ij,ij->i", filled, filled, dtype=np.float64, optimize=True)
        safe = np.maximum(counts, 1.0)
        mean = total / safe
        var = (sumsq - counts * mean * mean) / safe
        out[start:stop] = np.maximum(var, 0.0)
        del mask, filled
    return out


def full_variance_ranking(x_train_raw: np.ndarray, impl: str, var_block: int) -> np.ndarray:
    """Return column indices ordered by descending training-fold variance."""

    if impl == "nanvar":
        scores = np.nanvar(x_train_raw, axis=0)
    else:
        scores = fast_nanvar_axis0(x_train_raw, block=var_block)
    big = 1e300
    scores = np.nan_to_num(scores, nan=-big, posinf=big, neginf=-big)
    idx = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((idx, -scores)).astype(int)


def pairwise_abs_corr(x_imp: np.ndarray, block: int = 2000) -> np.ndarray:
    """Absolute Pearson correlation between columns of an imputed matrix.

    Computed in blocks so peak memory stays O(block * n_features) rather than
    O(n_features^2 * n_samples).  Constant columns are guarded instead of
    producing NaNs.
    """

    x = np.asarray(x_imp, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->j", x, x))
    norms = np.where(norms > 0, norms, 1e-12)
    x = x / norms
    n_features = x.shape[1]
    out = np.empty((n_features, n_features), dtype=np.float32)
    for start in range(0, n_features, block):
        stop = min(start + block, n_features)
        out[start:stop] = (x[:, start:stop].T @ x).astype(np.float32)
    return np.abs(out, out=out)


def mean_offdiag_abs(matrix: np.ndarray) -> float:
    k = matrix.shape[0]
    if k < 2:
        return float("nan")
    iu = np.triu_indices(k, k=1)
    return float(np.mean(matrix[iu]))


def select_dedup(abs_corr: np.ndarray, tau: float, k_target: int) -> tuple[np.ndarray, int]:
    """Greedy redundancy-controlled selection over a score-ordered pool.

    The pool arrives already ordered by descending marginal score, so walking
    ``range(n_pool)`` walks the ranking.  A probe is accepted when its maximum
    absolute correlation against the already-accepted probes is <= ``tau``.  If
    the filter cannot deliver ``k_target`` probes, the remainder is filled from
    the rejected probes in ascending order of their current maximum correlation,
    so that every arm is evaluated at an identical panel size.  ``n_dedup_clean``
    -- the number that passed the filter itself -- is returned as the honest
    measure of how many independent signals the pool contained.
    """

    n_pool = abs_corr.shape[0]
    max_corr = np.zeros(n_pool, dtype=np.float32)
    selected: list[int] = []
    for idx in range(n_pool):
        if max_corr[idx] <= tau:
            selected.append(idx)
            np.maximum(max_corr, abs_corr[idx], out=max_corr)
            if len(selected) >= k_target:
                break

    n_clean = len(selected)
    if n_clean < k_target:
        taken = np.zeros(n_pool, dtype=bool)
        taken[np.asarray(selected, dtype=int)] = True
        rest = np.flatnonzero(~taken)
        order = np.lexsort((rest, max_corr[rest]))
        filled = rest[order][: k_target - n_clean]
        selected = selected + [int(v) for v in filled]
    return np.asarray(selected, dtype=int), n_clean


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = list(ARMS) if args.arms.lower() == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    panel_sizes = sorted({int(v) for v in args.panel_sizes.split(",") if v.strip()})
    k_max = max(panel_sizes)
    sort_rows = not args.no_sort_rows

    splits_dir = Path(args.splits_dir)
    samples_df = pd.read_csv(splits_dir / "samples_locked.tsv", sep="\t")
    folds_df = pd.read_csv(splits_dir / "fold_assignments.tsv", sep="\t")
    split_summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    label_classes = list(split_summary["labels"])
    n_classes = len(label_classes)
    samples_lookup = samples_df.set_index("matrix_row")

    probe_ids = pd.read_csv(Path(args.probe_ids), sep="\t")["probe_id"].astype(str).tolist()
    if args.no_in_memory:
        x = np.load(args.matrix_npy, mmap_mode="r")
        matrix_mode = "memmap"
    else:
        x = np.load(args.matrix_npy)
        matrix_mode = "in_memory"
    if x.shape[1] != len(probe_ids):
        raise ValueError("probe id count mismatch")
    print(f"[data] matrix {x.shape} dtype={x.dtype} mode={matrix_mode}", flush=True)

    fold_index = fold_pairs(folds_df)
    if args.max_fold_runs > 0:
        fold_index = fold_index[: args.max_fold_runs]
    print(
        f"[plan] arms={arms} folds={len(fold_index)} panels={panel_sizes} "
        f"tau={args.corr_threshold} pool={args.prefilter} score_impl={args.score_impl} sort_rows={sort_rows}",
        flush=True,
    )

    accum = {
        arm: {
            "metric_rows": [],
            "redundancy_rows": [],
            "selected_rows": [],
            "prediction_rows": [],
        }
        for arm in arms
    }
    completed_run_ids: dict[str, set[str]] = {arm: set() for arm in arms}
    if args.resume:
        for arm in arms:
            metrics_path = out_dir / arm / "fold_metrics.tsv"
            if metrics_path.exists():
                old = pd.read_csv(metrics_path, sep="\t")
                if "run_id" in old.columns:
                    completed_run_ids[arm] = set(old["run_id"].astype(str))
                    accum[arm]["metric_rows"] = old.to_dict("records")
            oof_path = out_dir / arm / "oof_sample_predictions.tsv.gz"
            if oof_path.exists():
                accum[arm]["prediction_rows"].append(pd.read_csv(oof_path, sep="\t"))

    def flush_arm(arm: str) -> None:
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        store = accum[arm]
        if store["metric_rows"]:
            pd.DataFrame(store["metric_rows"]).to_csv(arm_dir / "fold_metrics.tsv", sep="\t", index=False)
        if store["redundancy_rows"]:
            pd.DataFrame(store["redundancy_rows"]).to_csv(arm_dir / "panel_redundancy.tsv", sep="\t", index=False)
        if store["prediction_rows"]:
            pd.concat(store["prediction_rows"], ignore_index=True).drop_duplicates(
                ["run_id", "matrix_row"], keep="last"
            ).to_csv(arm_dir / "oof_sample_predictions.tsv.gz", sep="\t", index=False, compression="gzip")
        if store["selected_rows"]:
            pd.DataFrame(store["selected_rows"]).to_csv(arm_dir / "selected_probe_manifest.tsv", sep="\t", index=False)

    for repeat, fold in fold_index:
        train_rows = subset_rows(folds_df, repeat, fold, "train")
        test_rows = subset_rows(folds_df, repeat, fold, "test")
        if sort_rows:
            train_rows = np.sort(train_rows)
            test_rows = np.sort(test_rows)
        y_train = labels_for_rows(samples_df, train_rows)
        y_test = labels_for_rows(samples_df, test_rows)

        pending = [
            arm
            for arm in arms
            if any(f"{arm}__r{repeat:02d}f{fold:02d}__top{ps}" not in completed_run_ids[arm] for ps in panel_sizes)
        ]
        if not pending:
            print(f"[skip] fold r{repeat} f{fold}: all arms complete", flush=True)
            continue

        fold_start = time.time()
        t = time.time()
        x_train_raw = np.asarray(x[train_rows], dtype=np.float32)
        x_test_raw = np.asarray(x[test_rows], dtype=np.float32)
        gather_seconds = time.time() - t

        t = time.time()
        order_full = full_variance_ranking(x_train_raw, args.score_impl, args.var_block)
        rank_seconds = time.time() - t

        arm_orders: dict[str, tuple[np.ndarray, int]] = {}
        dedup_seconds = 0.0
        if "variance_full" in pending:
            arm_orders["variance_full"] = (order_full[:k_max], -1)
        if "dedup" in pending:
            t = time.time()
            pool_global = order_full[: min(args.prefilter, x_train_raw.shape[1])]
            x_pool = x_train_raw[:, pool_global]
            x_pool_imp = SimpleImputer(strategy="median").fit_transform(x_pool)
            abs_corr = pairwise_abs_corr(x_pool_imp, block=args.corr_block)
            del x_pool, x_pool_imp
            local_selected, n_clean = select_dedup(abs_corr, tau=args.corr_threshold, k_target=k_max)
            del abs_corr
            arm_orders["dedup"] = (pool_global[local_selected], n_clean)
            dedup_seconds = time.time() - t

        print(
            f"[fold r{repeat} f{fold}] n_train={len(train_rows)} n_test={len(test_rows)} "
            f"gather={gather_seconds:.1f}s rank={rank_seconds:.1f}s dedup={dedup_seconds:.1f}s",
            flush=True,
        )

        for arm in pending:
            order, n_clean = arm_orders[arm]
            store = accum[arm]
            for ps in panel_sizes:
                run_id = f"{arm}__r{repeat:02d}f{fold:02d}__top{ps}"
                if run_id in completed_run_ids[arm]:
                    continue
                t = time.time()
                selected_idx = order[:ps]
                pipe = build_lr(args.seed + repeat * 100 + fold, args.n_jobs, args.max_iter)
                pipe.fit(x_train_raw[:, selected_idx], y_train)
                y_proba = full_grid_proba(pipe, x_test_raw[:, selected_idx], n_classes=n_classes)
                y_pred = y_proba.argmax(axis=1)
                fit_seconds = time.time() - t

                metrics = multiclass_metrics(y_test, y_pred, y_proba)
                cal = calibration_metrics(y_test, y_proba)
                store["metric_rows"].append(
                    {
                        "run_id": run_id,
                        "arm": arm,
                        "panel_size": ps,
                        "repeat": repeat,
                        "fold": fold,
                        "gather_seconds": float(gather_seconds),
                        "rank_seconds": float(rank_seconds),
                        "dedup_seconds": float(dedup_seconds),
                        "fit_seconds": float(fit_seconds),
                        **metrics,
                        **cal,
                    }
                )

                panel_imp = SimpleImputer(strategy="median").fit_transform(x_train_raw[:, selected_idx])
                panel_corr = pairwise_abs_corr(panel_imp, block=args.corr_block)
                store["redundancy_rows"].append(
                    {
                        "arm": arm,
                        "panel_size": ps,
                        "repeat": repeat,
                        "fold": fold,
                        "mean_abs_corr": mean_offdiag_abs(panel_corr),
                        "max_abs_corr": float(
                            np.max(panel_corr - np.eye(panel_corr.shape[0], dtype=np.float32))
                        ),
                        "n_dedup_clean": int(n_clean),
                        "pool_size": int(min(args.prefilter, len(order_full))),
                    }
                )
                del panel_imp, panel_corr

                pred_df = uncertainty_frame(y_test, y_proba, label_classes)
                pred_df.insert(0, "y_true", y_test)
                pred_df.insert(
                    0,
                    "q1_patient_id",
                    samples_lookup.loc[test_rows, "q1_patient_id"].astype(str).to_numpy(),
                )
                pred_df.insert(0, "fold", fold)
                pred_df.insert(0, "repeat", repeat)
                pred_df.insert(0, "panel_size", ps)
                pred_df.insert(0, "matrix_row", test_rows)
                pred_df.insert(0, "run_id", run_id)
                for cid in range(n_classes):
                    pred_df[f"probability_class_{cid}"] = y_proba[:, cid]
                store["prediction_rows"].append(pred_df)

                for rank, pid in enumerate(selected_idx, start=1):
                    store["selected_rows"].append(
                        {
                            "arm": arm,
                            "panel_size": ps,
                            "repeat": repeat,
                            "fold": fold,
                            "rank": rank,
                            "probe_index": int(pid),
                            "probe_id": probe_ids[int(pid)],
                        }
                    )
                completed_run_ids[arm].add(run_id)

            flush_arm(arm)
            print(f"[{arm}] fold r{repeat} f{fold} checkpointed", flush=True)

        print(f"[fold r{repeat} f{fold}] total {time.time() - fold_start:.1f}s", flush=True)
        del x_train_raw, x_test_raw, order_full, arm_orders

    for arm in arms:
        arm_dir = out_dir / arm
        store = accum[arm]
        if not store["selected_rows"]:
            print(f"[{arm}] no selections recorded", flush=True)
            continue
        flush_arm(arm)

        sel = pd.DataFrame(store["selected_rows"])
        stab_rows: list[dict] = []
        for ps in panel_sizes:
            folds_ps = sel[sel["panel_size"] == ps]
            by_fold = {key: set(g["probe_index"]) for key, g in folds_ps.groupby(["repeat", "fold"])}
            keys = sorted(by_fold)
            js: list[float] = []
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    union = len(by_fold[a] | by_fold[b])
                    if union:
                        js.append(len(by_fold[a] & by_fold[b]) / union)
            if js:
                stab_rows.append(
                    {
                        "arm": arm,
                        "panel_size": ps,
                        "n_fold_pairs": len(keys) * (len(keys) - 1) // 2,
                        "mean_pairwise_jaccard": float(np.mean(js)),
                        "min": float(np.min(js)),
                        "max": float(np.max(js)),
                    }
                )
        if stab_rows:
            pd.DataFrame(stab_rows).to_csv(arm_dir / "selection_stability.tsv", sep="\t", index=False)

        oof_path = arm_dir / "oof_sample_predictions.tsv.gz"
        evaluate_repeated_oof(oof_path, splits_dir / "split_summary.json", arm_dir, n_bootstrap=2000, seed=args.seed)

        manifest = {
            "experiment": "E6_redundancy_aware_selection_pilot",
            "arm": arm,
            "score_statistic": "training-fold variance",
            "score_impl": args.score_impl,
            "downstream_model": "LR (imputer median + scaler std-only + lbfgs balanced)",
            "selection_space": "full candidate space" if arm == "variance_full" else f"outer-training variance top-{args.prefilter} pool",
            "corr_threshold": args.corr_threshold,
            "panel_sizes": panel_sizes,
            "seed": args.seed,
            "n_folds": len(fold_index),
            "row_order": "sorted" if sort_rows else "fold_shuffled",
            "leakage_contract": "all scoring, pooling, de-duplication and fitting inside the outer-training partition",
            "scope_note": "pilot: single redundancy threshold, no threshold sweep",
        }
        (arm_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{arm}] done", flush=True)

    parity: list[pd.DataFrame] = []
    for arm in arms:
        p = out_dir / arm / "panel_redundancy.tsv"
        if p.exists():
            parity.append(pd.read_csv(p, sep="\t"))
    if parity:
        merged = pd.concat(parity, ignore_index=True)
        summary = (
            merged.groupby(["arm", "panel_size"], as_index=False)
            .agg(
                mean_abs_corr=("mean_abs_corr", "mean"),
                max_abs_corr=("max_abs_corr", "mean"),
                n_dedup_clean=("n_dedup_clean", "mean"),
                pool_size=("pool_size", "mean"),
            )
        )
        stability_frames = []
        for arm in arms:
            sp = out_dir / arm / "selection_stability.tsv"
            if sp.exists():
                stability_frames.append(pd.read_csv(sp, sep="\t"))
        if stability_frames:
            stab = pd.concat(stability_frames, ignore_index=True)[["arm", "panel_size", "mean_pairwise_jaccard"]]
            summary = summary.merge(stab, on=["arm", "panel_size"], how="left")
        summary.to_csv(out_dir / "parity_summary.tsv", sep="\t", index=False)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
