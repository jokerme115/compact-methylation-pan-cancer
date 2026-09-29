"""E7 part 2: geometry of the variance-ranked panel ladder.

Question
--------
Discrimination saturates around 1,500 CpGs while calibration keeps degrading
up to 10,000 (see ``run_revision_e7_panel_size_paired_bootstrap.py``).  Why do
the extra 8,500 CpGs buy nothing?

This script measures the two candidate explanations on the same folds the E1
ladder used, entirely in-fold (no held-out data touched, no external cohort):

1. **Effective dimension.**  For every panel size ``k`` we eigendecompose the
   training-fold correlation matrix of the top-``k`` variance-ranked CpGs and
   report how many principal components are needed to explain 90% / 95% of the
   total variance, plus the participation ratio.  If ``n_pc_95`` grows much
   slower than ``k``, the panel is not adding independent directions.

2. **Tail redundancy.**  Split the top-10,000 panel into a head (top 1,500) and
   a tail (ranks 1,501-10,000).  We then ask what fraction of the tail's total
   variance lies inside the head panel's principal subspace, and vice versa.
   A high ``r2_tail_in_head`` means the tail is *redundant* (its information is
   already spanned by the head) rather than merely uninformative.  The maximum
   principal cosine between the two subspaces is reported alongside.

Outputs (under ``--out-dir``)
-----------------------------
- ``panel_eigen_summary.tsv``     one row per (repeat, fold, panel_size)
- ``panel_eigen_summary_mean.tsv`` fold-aggregated, with marginal PC gain
- ``subspace_overlap.tsv``        one row per (repeat, fold)
- ``run_manifest.json``

Cost note: the dominant terms are the 10,000 x 10,000 Gram matrix and its
eigendecomposition.  Run this on a subset of folds with ``--repeats`` /
``--max-fold-runs`` when validating the code path.
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

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/data", "src/features", "src/models", "src/evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from q1_train_ml_baselines import fold_pairs, subset_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E7 panel geometry: effective dimension + tail redundancy.")
    p.add_argument("--matrix-npy", required=True)
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--panel-sizes", default="500,1000,1500,2000,5000,10000")
    p.add_argument("--k-max", type=int, default=10000, help="Size of the candidate window to decompose.")
    p.add_argument("--head-size", type=int, default=1500, help="Head panel size for the head/tail split.")
    p.add_argument(
        "--repeats",
        default="0",
        help="Comma-separated repeats to run. Default '0': the panels are ~97%% identical across "
        "folds and repeats, so one repeat characterises the spectrum.",
    )
    p.add_argument("--max-fold-runs", type=int, default=0, help="Limit folds per repeat (smoke only).")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def variance_ranking(x_train_raw: np.ndarray) -> np.ndarray:
    """Column indices ordered by descending training-fold variance (E1 recipe)."""

    scores = np.nanvar(x_train_raw, axis=0)
    big = 1e300
    scores = np.nan_to_num(scores, nan=-big, posinf=big, neginf=-big)
    idx = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((idx, -scores)).astype(int)


def spectrum_stats(eigenvalues: np.ndarray) -> dict[str, float]:
    """Summarise the spectrum of a correlation matrix.

    ``numpy.linalg.eigvalsh``/``eigh`` return eigenvalues in **ascending**
    order, so the cumulative variance has to be formed from the largest end;
    summing from the smallest would always report ``n_pc_90 == k``.
    """

    ev = np.sort(eigenvalues[eigenvalues > 0])[::-1]
    if ev.size == 0:
        return {
            "n_pc_90": np.nan,
            "n_pc_95": np.nan,
            "participation_ratio": np.nan,
            "top1_share": np.nan,
        }
    total = float(ev.sum())
    cumulative = np.cumsum(ev)
    n_pc_90 = int(np.searchsorted(cumulative, 0.90 * total) + 1)
    n_pc_95 = int(np.searchsorted(cumulative, 0.95 * total) + 1)
    pr = float(total * total / float(np.sum(ev * ev)))
    return {
        "n_pc_90": n_pc_90,
        "n_pc_95": n_pc_95,
        "participation_ratio": pr,
        "top1_share": float(ev[0] / total),
    }


def standardise(x: np.ndarray) -> np.ndarray:
    """Median-impute then z-score each column; returns float32."""

    x_imp = SimpleImputer(strategy="median").fit_transform(x)
    x_imp = np.asarray(x_imp, dtype=np.float32)
    mean = x_imp.mean(axis=0, keepdims=True)
    std = x_imp.std(axis=0, keepdims=True)
    std = np.where(std > 0, std, np.float32(1.0))
    return (x_imp - mean) / std


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(v) for v in args.panel_sizes.split(",") if v.strip()})
    repeats_wanted = {int(v) for v in args.repeats.split(",") if v.strip()}
    k_max = int(args.k_max)
    head = int(args.head_size)
    if head >= k_max:
        raise ValueError("--head-size must be smaller than --k-max")
    if max(sizes) > k_max:
        raise ValueError("--panel-sizes must not exceed --k-max")

    splits_dir = Path(args.splits_dir)
    folds_df = pd.read_csv(splits_dir / "fold_assignments.tsv", sep="\t")
    fold_index = [(r, f) for r, f in fold_pairs(folds_df) if r in repeats_wanted]
    if args.max_fold_runs > 0:
        by_repeat: dict[int, list[tuple[int, int]]] = {}
        for r, f in fold_index:
            by_repeat.setdefault(r, []).append((r, f))
        fold_index = [t for r in sorted(by_repeat) for t in by_repeat[r][: args.max_fold_runs]]

    eigen_path = out_dir / "panel_eigen_summary.tsv"
    overlap_path = out_dir / "subspace_overlap.tsv"
    eigen_rows: list[dict] = []
    overlap_rows: list[dict] = []
    done: set[tuple[int, int]] = set()
    if args.resume and eigen_path.exists():
        old = pd.read_csv(eigen_path, sep="\t")
        eigen_rows = old.to_dict("records")
        done = {(int(r), int(f)) for r, f in zip(old["repeat"], old["fold"])}
        if overlap_path.exists():
            overlap_rows = pd.read_csv(overlap_path, sep="\t").to_dict("records")

    x = np.load(args.matrix_npy)
    print(f"[data] matrix {x.shape} dtype={x.dtype}", flush=True)
    print(f"[plan] folds={len(fold_index)} sizes={sizes} k_max={k_max} head={head}", flush=True)

    for repeat, fold in fold_index:
        if (repeat, fold) in done:
            print(f"[skip] fold r{repeat} f{fold}", flush=True)
            continue
        t_fold = time.time()
        train_rows = np.sort(subset_rows(folds_df, repeat, fold, "train"))
        t = time.time()
        x_train_raw = np.asarray(x[train_rows], dtype=np.float32)
        gather_s = time.time() - t

        t = time.time()
        order = variance_ranking(x_train_raw)[:k_max]
        rank_s = time.time() - t

        t = time.time()
        xc = standardise(x_train_raw[:, order])
        del x_train_raw
        n_samples = xc.shape[0]
        xc64 = np.asarray(xc, dtype=np.float64)
        corr = (xc64.T @ xc64) / max(n_samples - 1, 1)
        corr = (corr + corr.T) * 0.5
        gram_s = time.time() - t

        t = time.time()
        eigen_s: list[float] = 0.0
        for k in sizes:
            t_k = time.time()
            block = np.ascontiguousarray(corr[:k, :k])
            ev = np.linalg.eigvalsh(block)
            stats = spectrum_stats(ev)
            eigen_s += time.time() - t_k
            eigen_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "panel_size": k,
                    "n_samples_train": n_samples,
                    "trace": float(ev.sum()),
                    **stats,
                }
            )
            del block, ev
            print(
                f"[eig] r{repeat} f{fold} k={k} n_pc_90={stats['n_pc_90']} "
                f"n_pc_95={stats['n_pc_95']} ({time.time() - t_k:.1f}s)",
                flush=True,
            )

        # Head / tail subspace overlap inside the top-k_max window.
        t_sub = time.time()
        head_block = np.ascontiguousarray(corr[:head, :head])
        lam_h, vec_h = np.linalg.eigh(head_block)
        del head_block
        stats_head = spectrum_stats(lam_h)
        r_head = max(int(stats_head["n_pc_95"]), 1)
        # largest eigenvalues/vectors come last in numpy's ascending order
        top_h = slice(len(lam_h) - r_head, len(lam_h))
        q_head = (xc[:, :head] @ vec_h[:, top_h]) / np.sqrt(np.maximum(lam_h[top_h], 1e-12))
        q_head = np.linalg.qr(q_head)[0]

        tail_block = np.ascontiguousarray(corr[head:k_max, head:k_max])
        lam_t, vec_t = np.linalg.eigh(tail_block)
        del tail_block
        stats_tail = spectrum_stats(lam_t)
        r_tail = max(int(stats_tail["n_pc_95"]), 1)

        # Same-rank comparison so the two directions are directly readable: the
        # tail basis uses `r_head` components, mirroring the head basis.
        q_tail_r = np.linalg.qr(xc[:, head:k_max] @ vec_t[:, len(lam_t) - r_head :])[0]

        xc_head = xc[:, :head]
        xc_tail = xc[:, head:k_max]
        denom_head = float(np.einsum("ij,ij->", xc_head, xc_head))
        denom_tail = float(np.einsum("ij,ij->", xc_tail, xc_tail))

        proj_head = q_head.T @ xc_tail
        r2_tail_in_head = float(np.einsum("ij,ij->", proj_head, proj_head) / denom_tail)
        del proj_head

        proj_tail = q_tail_r.T @ xc_head
        r2_head_in_tail = float(np.einsum("ij,ij->", proj_tail, proj_tail) / denom_head)
        del proj_tail

        cosines = np.linalg.svd(q_head.T @ q_tail_r, compute_uv=False)
        overlap_rows.append(
            {
                "repeat": repeat,
                "fold": fold,
                "n_samples_train": n_samples,
                "k_head": head,
                "k_tail": k_max - head,
                "k_max": k_max,
                "n_pc_95_head": r_head,
                "n_pc_95_tail": r_tail,
                "r2_tail_in_head": r2_tail_in_head,
                "r2_head_in_tail": r2_head_in_tail,
                "max_principal_cosine": float(np.max(cosines)),
                "mean_principal_cosine": float(np.mean(cosines)),
            }
        )
        del q_head, q_tail_r, xc_head, xc_tail, xc64, corr, xc, lam_h, vec_h, lam_t, vec_t
        sub_s = time.time() - t_sub

        pd.DataFrame(eigen_rows).to_csv(out_dir / "panel_eigen_summary.tsv", sep="\t", index=False)
        pd.DataFrame(overlap_rows).to_csv(out_dir / "subspace_overlap.tsv", sep="\t", index=False)
        print(
            f"[fold r{repeat} f{fold}] gather={gather_s:.1f}s rank={rank_s:.1f}s gram={gram_s:.1f}s "
            f"eig={eigen_s:.1f}s subspace={sub_s:.1f}s total={time.time() - t_fold:.1f}s",
            flush=True,
        )

    eigen_df = pd.DataFrame(eigen_rows)
    t = time.time()
    agg = (
        eigen_df.groupby("panel_size")
        .agg(
            n_pc_90=("n_pc_90", "mean"),
            n_pc_95=("n_pc_95", "mean"),
            n_pc_95_sd=("n_pc_95", "std"),
            participation_ratio=("participation_ratio", "mean"),
            top1_share=("top1_share", "mean"),
            n_folds=("fold", "count"),
        )
        .reset_index()
        .sort_values("panel_size")
    )
    agg["delta_k"] = agg["panel_size"].diff()
    agg["delta_n_pc_95"] = agg["n_pc_95"].diff()
    agg["marginal_pc_gain"] = agg["delta_n_pc_95"] / agg["delta_k"]
    agg["n_pc_95_per_k"] = agg["n_pc_95"] / agg["panel_size"]
    agg.to_csv(out_dir / "panel_eigen_summary_mean.tsv", sep="\t", index=False)

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(overlap_path, sep="\t", index=False)

    manifest = {
        "experiment": "E7_panel_geometry",
        "matrix": args.matrix_npy,
        "splits": args.splits_dir,
        "panel_sizes": sizes,
        "k_max": k_max,
        "head_size": head,
        "repeats": sorted(repeats_wanted),
        "ranking": "training-fold variance (np.nanvar), identical to the E1 ladder",
        "preprocessing": "median impute + column z-score, computed inside the training fold only",
        "spectrum": "eigenvalues of the top-k training-fold correlation matrix (k x k prefix block)",
        "r2_tail_in_head": "mean squared projection of the standardised tail columns onto the "
                            "head panel's top-r_head principal subspace, divided by the tail's total "
                            "sum of squares; r_head = n_pc_95 of the head block",
        "r2_head_in_tail": "the mirror quantity with r_head components taken from the tail block, "
                           "so both directions use the same rank",
        "note": "In-fold geometry only; no external cohort, no held-out data. "
                "Exploratory revision-stage evidence.",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[E7geom] done -> {out_dir} ({time.time() - t:.1f}s aggregation)", flush=True)


if __name__ == "__main__":
    main()
