#!/usr/bin/env python3
"""Compare two fresh v5 anchor reruns with the 16-panel v5 reference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PANELS = [1000, 1500, 2000]
PROB = [f"probability_class_{i}" for i in range(33)]
KEY = ["panel_size", "repeat", "q1_patient_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="v5 full-run result directory")
    p.add_argument("--run-a", required=True, help="fresh anchor run A result directory")
    p.add_argument("--run-b", required=True, help="fresh anchor run B result directory")
    p.add_argument("--output", required=True)
    p.add_argument("--prob-tol", type=float, default=1e-8)
    p.add_argument("--metric-tol", type=float, default=1e-6)
    return p.parse_args()


def load_patient(run: Path) -> pd.DataFrame:
    use = KEY + ["y_true", "y_pred"] + PROB
    d = pd.read_csv(run / "patient_oof_predictions_by_repeat.tsv.gz", sep="\t", usecols=use)
    return d[d.panel_size.isin(PANELS)].sort_values(KEY).reset_index(drop=True)


def primary_macro_f1(d: pd.DataFrame) -> dict[int, float]:
    rows = []
    for (panel, repeat), g in d.groupby(["panel_size", "repeat"], sort=True):
        rows.append((int(panel), int(repeat), f1_score(g.y_true, g.y_pred, average="macro")))
    x = pd.DataFrame(rows, columns=["panel_size", "repeat", "macro_f1"])
    return {int(k): float(v) for k, v in x.groupby("panel_size").macro_f1.mean().items()}


def compare(name: str, left: pd.DataFrame, right: pd.DataFrame, prob_tol: float, metric_tol: float) -> dict:
    result = {"comparison": name, "keys_equal": bool(left[KEY].equals(right[KEY]))}
    if not result["keys_equal"]:
        result["pass"] = False
        return result
    delta = np.abs(left[PROB].to_numpy() - right[PROB].to_numpy())
    lm = primary_macro_f1(left)
    rm = primary_macro_f1(right)
    result.update(
        {
            "y_true_equal": bool(np.array_equal(left.y_true, right.y_true)),
            "prediction_mismatches": int((left.y_pred.to_numpy() != right.y_pred.to_numpy()).sum()),
            "max_probability_difference": float(delta.max()),
            "macro_f1_differences": {str(p): float(lm[p] - rm[p]) for p in PANELS},
        }
    )
    result["pass"] = bool(
        result["y_true_equal"]
        and result["prediction_mismatches"] == 0
        and result["max_probability_difference"] <= prob_tol
        and max(abs(v) for v in result["macro_f1_differences"].values()) <= metric_tol
    )
    return result


def check_convergence(run: Path) -> dict:
    p = run / "convergence_table.tsv"
    if not p.exists():
        return {"exists": False, "pass": False}
    d = pd.read_csv(p, sep="\t")
    d = d[d.panel_size.isin(PANELS)]
    flags = d.converged.astype(str).str.lower().eq("true")
    ok = len(d) == 45 and d.run_id.nunique() == 45 and flags.all() and (d.n_iter_max < d.max_iter).all()
    return {
        "exists": True,
        "rows": int(len(d)),
        "all_converged": bool(flags.all()),
        "n_iter_max": int(d.n_iter_max.max()),
        "pass": bool(ok),
    }


def selected_prefixes(run: Path) -> dict[tuple[int, int], list[str]]:
    out = {}
    for p in sorted((run / "selected_features").glob("selected_top*_r*f*.tsv")):
        d = pd.read_csv(p, sep="\t")
        out[(int(d.repeat.iloc[0]), int(d.fold.iloc[0]))] = d.probe_id.astype(str).head(2000).tolist()
    return out


def compare_rankings(name: str, left: Path, right: Path) -> dict:
    a = selected_prefixes(left)
    b = selected_prefixes(right)
    cells = sorted(set(a) | set(b))
    mismatches = {f"r{r:02d}f{f:02d}": None if (r, f) not in a or (r, f) not in b else sum(x != y for x, y in zip(a[(r, f)], b[(r, f)])) for r, f in cells}
    ok = len(cells) == 15 and all(v == 0 for v in mismatches.values())
    return {"comparison": name, "cells": len(cells), "position_mismatches": mismatches, "pass": bool(ok)}


def main() -> None:
    args = parse_args()
    ref, a, b = map(Path, [args.reference, args.run_a, args.run_b])
    rp, ap, bp = map(load_patient, [ref, a, b])
    comparisons = [
        compare("A_vs_B", ap, bp, args.prob_tol, args.metric_tol),
        compare("A_vs_reference", ap, rp, args.prob_tol, args.metric_tol),
        compare("B_vs_reference", bp, rp, args.prob_tol, args.metric_tol),
    ]
    rankings = [
        compare_rankings("A_vs_B", a, b),
        compare_rankings("A_vs_reference", a, ref),
        compare_rankings("B_vs_reference", b, ref),
    ]
    convergence = {"run_a": check_convergence(a), "run_b": check_convergence(b)}
    ab_pass = comparisons[0]["pass"] and rankings[0]["pass"] and convergence["run_a"]["pass"] and convergence["run_b"]["pass"]
    reference_pass = ab_pass and all(x["pass"] for x in comparisons[1:]) and all(x["pass"] for x in rankings[1:])
    verdict = "PASS_REFERENCE" if reference_pass else "PASS_AB_ONLY" if ab_pass else "FAIL_AB"
    result = {
        "verdict": verdict,
        "thresholds": {"probability": args.prob_tol, "macro_f1": args.metric_tol},
        "comparisons": comparisons,
        "rankings": rankings,
        "convergence": convergence,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if verdict == "PASS_REFERENCE" else 2)


if __name__ == "__main__":
    main()
