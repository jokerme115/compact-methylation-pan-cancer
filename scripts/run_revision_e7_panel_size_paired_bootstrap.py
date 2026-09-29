"""E7: paired panel-size comparison of the frozen E1 panel-size ladder.

Motivation
----------
The E1 ladder (``results/revision_rerun_20260907/E1_panel_knee_v5_anchor_verify``)
evaluates 16 nested variance-ranked panels with a patient-level repeated CV.
Discrimination saturates around 1,500 CpGs, but the calibration metrics
(``nll``, ``ece_15_bins``) do not -- they keep worsening as the panel grows.
This script turns those marginal CIs into **paired** ones: both panels of a
pair see exactly the same folds, the same patients and the same repeat
structure, so the comparison has far more power than reading two overlapping
marginal intervals side by side.

Input
-----
``patient_oof_predictions_by_repeat.tsv.gz`` from the E1 final package.  One
row per ``(panel_size, repeat, q1_patient_id)`` carrying the 33 class
probabilities, ``y_true`` and the aggregated ``y_pred``.  No model is refitted
-- this is a pure re-analysis of already frozen predictions.

Protocol (identical to ``run_revision_e2_paired_bootstrap.py``)
--------------------------------------------------------------
1. For every repeat, compute the patient-level metric for panel A and panel B.
2. ``delta = metric(B) - metric(A)``, averaged over repeats (repeat-specific
   deltas are averaged, not metrics averaged then differenced).
3. Bootstrap: draw patients with replacement **once per iteration** and apply
   that same index vector to every repeat and to both panels, so the pairing is
   preserved.  2000 iterations, percentile 2.5 / 97.5 interval.

Output
------
``panel_size_paired_bootstrap.tsv`` (long format: one row per pair x metric)
plus ``run_manifest.json``.

Exploratory revision-stage evidence only.  ``fraction_gt_zero`` is NOT a
p-value: it is the share of resamples with the same sign as the point estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/data", "src/features", "src/models", "src/evaluation"):
    sys.path.insert(0, str(ROOT / sub))

from q1_metrics import expected_calibration_error, multiclass_brier  # noqa: E402

# Metrics computed per (panel, repeat).  Definitions come from ``q1_metrics``
# wherever they exist there, so the numbers are directly comparable with the
# frozen ``summary_metrics_by_panel.csv``.
METRICS = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "nll",
    "brier_multiclass",
    "ece_15_bins",
    "aurc",
)
LOWER_IS_BETTER = {"nll", "brier_multiclass", "ece_15_bins", "aurc"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E7 paired panel-size comparison (E1 ladder).")
    p.add_argument("--pred-file", required=True, help="patient_oof_predictions_by_repeat.tsv.gz")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--panel-pairs",
        default="500:1000,1000:1500,500:1500,1500:2000,1500:5000,1500:10000,1000:10000,500:10000",
        help="Comma-separated 'a:b' pairs; the delta is reported as b - a.",
    )
    p.add_argument("--metrics", default=",".join(METRICS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


def patient_metrics(yt: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Patient-level metrics from a probability matrix.

    ``nll`` is computed as ``-mean(log p_true)``, which is algebraically the
    ``sklearn.metrics.log_loss`` value for a normalised multiclass probability
    matrix but a few hundred times cheaper per bootstrap iteration.
    """

    proba = np.asarray(proba, dtype=float)
    yp = proba.argmax(axis=1)
    true_prob = np.clip(proba[np.arange(len(yt)), yt], 1e-15, 1.0)
    confidence = proba.max(axis=1)
    wrong = (yp != yt).astype(float)
    order = np.argsort(-confidence, kind="mergesort")
    aurc_curve = np.cumsum(wrong[order]) / np.arange(1, len(wrong) + 1)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "nll": float(-np.mean(np.log(true_prob))),
        "brier_multiclass": multiclass_brier(yt, proba),
        "ece_15_bins": expected_calibration_error(yt, proba, n_bins=15),
        "aurc": float(aurc_curve.mean()),
    }


def load_predictions(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[int], list[int]]:
    """Return per-panel ``y_true`` and probability tensors indexed by repeat.

    The returned arrays have shape ``(n_repeats, n_patients)`` for the labels
    and ``(n_repeats, n_patients, n_classes)`` for the probabilities.
    """

    df = pd.read_csv(path, sep="\t", compression="gzip")
    prob_cols = sorted(
        (c for c in df.columns if c.startswith("probability_class_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not prob_cols:
        raise ValueError("no probability_class_* columns found")
    df["q1_patient_id"] = df["q1_patient_id"].astype(str)

    patients = sorted(df["q1_patient_id"].unique())
    repeats = sorted(int(v) for v in df["repeat"].unique())
    panels = sorted(int(v) for v in df["panel_size"].unique())
    pat_index = {p: i for i, p in enumerate(patients)}
    rep_index = {r: i for i, r in enumerate(repeats)}

    shape = (len(repeats), len(patients), len(prob_cols))
    y_true: dict[int, np.ndarray] = {}
    proba: dict[int, np.ndarray] = {}
    for panel in panels:
        block = df[df["panel_size"] == panel]
        yt = np.zeros((len(repeats), len(patients)), dtype=np.int16)
        pr = np.zeros(shape, dtype=np.float32)
        rows = block["q1_patient_id"].map(pat_index).to_numpy()
        reps = block["repeat"].astype(int).map(rep_index).to_numpy()
        yt[reps, rows] = block["y_true"].to_numpy(dtype=np.int16)
        pr[reps, rows, :] = block[prob_cols].to_numpy(dtype=np.float32)
        y_true[panel] = yt
        proba[panel] = pr
    return y_true, proba, patients, repeats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}; expected one of {METRICS}")

    pairs: list[tuple[int, int]] = []
    for token in args.panel_pairs.split(","):
        token = token.strip()
        if not token:
            continue
        a, b = token.split(":")
        pairs.append((int(a), int(b)))

    t0 = time.time()
    y_true, proba, patients, repeats = load_predictions(Path(args.pred_file))
    n_repeats = len(repeats)
    n_patients = len(patients)
    print(
        f"[data] panels={sorted(proba)} repeats={n_repeats} patients={n_patients} "
        f"load={time.time() - t0:.1f}s",
        flush=True,
    )

    rows: list[dict] = []
    for panel_a, panel_b in pairs:
        if panel_a not in proba or panel_b not in proba:
            raise ValueError(f"panel pair {panel_a}:{panel_b} not present in the file")
        yt = y_true[panel_a]
        if not np.array_equal(yt, y_true[panel_b]):
            raise RuntimeError(f"panel {panel_a}/{panel_b}: y_true mismatch")
        pt_a = proba[panel_a]
        pt_b = proba[panel_b]

        point = {m: [] for m in metrics}
        abs_a = {m: [] for m in metrics}
        abs_b = {m: [] for m in metrics}
        for ri in range(n_repeats):
            ma = patient_metrics(yt[ri], pt_a[ri])
            mb = patient_metrics(yt[ri], pt_b[ri])
            for m in metrics:
                point[m].append(mb[m] - ma[m])
                abs_a[m].append(ma[m])
                abs_b[m].append(mb[m])

        boot = {m: np.empty(args.n_bootstrap) for m in metrics}
        for bi in range(args.n_bootstrap):
            # One patient draw per iteration, shared by every repeat and both
            # panels -- that is what keeps the comparison paired.
            idx = rng.integers(0, n_patients, size=n_patients)
            acc = {m: np.zeros(n_repeats) for m in metrics}
            for ri in range(n_repeats):
                ma = patient_metrics(yt[ri][idx], pt_a[ri][idx])
                mb = patient_metrics(yt[ri][idx], pt_b[ri][idx])
                for m in metrics:
                    acc[m][ri] = mb[m] - ma[m]
            for m in metrics:
                boot[m][bi] = float(np.mean(acc[m]))

        for m in metrics:
            lo = float(np.percentile(boot[m], 2.5))
            hi = float(np.percentile(boot[m], 97.5))
            rows.append(
                {
                    "panel_a": panel_a,
                    "panel_b": panel_b,
                    "comparison": f"{panel_b}_minus_{panel_a}",
                    "metric": m,
                    "lower_is_better": m in LOWER_IS_BETTER,
                    "a_value": round(float(np.mean(abs_a[m])), 6),
                    "b_value": round(float(np.mean(abs_b[m])), 6),
                    "delta": round(float(np.mean(point[m])), 6),
                    "ci_low": round(lo, 6),
                    "ci_high": round(hi, 6),
                    "crosses_zero": bool(lo <= 0.0 <= hi),
                    "fraction_gt_zero": round(float((boot[m] > 0).mean()), 4),
                    "n_patients": n_patients,
                    "n_repeats": n_repeats,
                    "n_bootstrap": args.n_bootstrap,
                }
            )
            print(
                f"[{panel_a}->{panel_b}] {m:18s} delta={np.mean(point[m]):+.6f} "
                f"CI=[{lo:+.6f},{hi:+.6f}] zero={'YES' if lo <= 0 <= hi else 'no '}",
                flush=True,
            )

    df_wide = pd.DataFrame(rows)
    tmp_path = out_dir / ".panel_size_paired_bootstrap.tsv.tmp"
    df_wide.to_csv(tmp_path, sep="\t", index=False)
    os.replace(str(tmp_path), str(out_dir / "panel_size_paired_bootstrap.tsv"))

    # A compact pivot for reading: one column per metric, delta with CI.
    pivot = df_wide.copy()
    pivot["delta_with_ci"] = pivot.apply(
        lambda r: f"{r['delta']:+.5f} [{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]", axis=1
    )
    pivot.pivot_table(index="comparison", columns="metric", values="delta_with_ci", aggfunc="first").to_csv(
        out_dir / "panel_size_paired_bootstrap_wide.tsv", sep="\t"
    )

    manifest = {
        "experiment": "E7_panel_size_paired_bootstrap",
        "source": args.pred_file,
        "estimator": "mean over repeats of the repeat-specific patient-level metric delta",
        "bootstrap": "patient-clustered; one patient draw applied to all repeats and both panels",
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "panel_pairs": [f"{a}:{b}" for a, b in pairs],
        "metrics": metrics,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "crosses_zero": {
            f"{r['comparison']}|{r['metric']}": bool(r["crosses_zero"]) for r in rows
        },
        "note": (
            "Re-analysis of frozen E1 predictions; no model refitted. "
            "fraction_gt_zero is not a p-value. Exploratory revision-stage evidence."
        ),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[E7paired] done in {time.time() - t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
