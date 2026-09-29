#!/usr/bin/env python3
"""Compute submission-facing confidence intervals and paired panel comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
PANEL_ORDER = [100, 200, 500, 1000]
FORMAL_COHORTS = ["GSE56044", "GSE48684", "GSE53051", "GSE105260_official_series"]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def internal_statistics(outdir: Path) -> None:
    source = ROOT / "results/internal_cv/lr_cpg_panel/fold_metrics.tsv"
    df = pd.read_csv(source, sep="\t")
    df = df[df.panel_size.isin(PANEL_ORDER)].copy()

    rows = []
    rng = np.random.default_rng(20260710)
    for panel, group in df.groupby("panel_size", sort=True):
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "ece_15_bins"]:
            values = group[metric].dropna().to_numpy()
            lo, hi = bootstrap_ci(values, rng, 20000)
            rows.append({
                "panel_size": panel,
                "metric": metric,
                "n_folds": values.size,
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "bootstrap_95ci_low": lo,
                "bootstrap_95ci_high": hi,
            })
    pd.DataFrame(rows).to_csv(outdir / "internal_cv_bootstrap_ci.tsv", sep="\t", index=False)

    paired = []
    key = ["repeat", "fold"]
    for small, large in zip(PANEL_ORDER[:-1], PANEL_ORDER[1:]):
        left = df[df.panel_size == small].set_index(key)
        right = df[df.panel_size == large].set_index(key)
        for metric in ["accuracy", "macro_f1", "top3_accuracy", "ece_15_bins"]:
            delta = right[metric] - left[metric]
            test = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
            paired.append({
                "comparison": f"{large} vs {small}",
                "metric": metric,
                "n_paired_folds": delta.notna().sum(),
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "wilcoxon_statistic": test.statistic,
                "p_value": test.pvalue,
            })
    paired_df = pd.DataFrame(paired)
    paired_df["holm_significant_0_05"] = False
    paired_df["holm_adjusted_p_value"] = np.nan
    for metric, idx in paired_df.groupby("metric").groups.items():
        ordered = paired_df.loc[idx].sort_values("p_value")
        m = len(ordered)
        adjusted = []
        running_max = 0.0
        for rank, (row_idx, row) in enumerate(ordered.iterrows(), start=1):
            adjusted_value = min(1.0, (m - rank + 1) * float(row.p_value))
            running_max = max(running_max, adjusted_value)
            adjusted.append((row_idx, running_max))
        for row_idx, adjusted_value in adjusted:
            paired_df.loc[row_idx, "holm_adjusted_p_value"] = adjusted_value
        keep = []
        still_rejecting = True
        for rank, (row_idx, row) in enumerate(ordered.iterrows(), start=1):
            reject = still_rejecting and row.p_value <= 0.05 / (m - rank + 1)
            keep.append((row_idx, reject))
            still_rejecting = reject
        for row_idx, reject in keep:
            paired_df.loc[row_idx, "holm_significant_0_05"] = reject
    paired_df.to_csv(outdir / "internal_cv_paired_panel_tests.tsv", sep="\t", index=False)


def exact_ci(successes: int, total: int) -> tuple[float, float]:
    ci = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="exact")
    return ci.low, ci.high


def external_statistics(outdir: Path, n_boot: int) -> None:
    rng = np.random.default_rng(20260710)
    ci_rows = []
    comparison_rows = []
    for cohort in FORMAL_COHORTS:
        path = ROOT / f"results/external_validation/{cohort}/lr_panel/external_predictions.tsv"
        pred = pd.read_csv(path, sep="\t")
        for panel, group in pred.groupby("panel_size", sort=True):
            correct = group.accepted_correct.astype(bool).to_numpy()
            top3 = np.array([
                any(label in set(str(accepted).split(";")) for label in [r.top1_label, r.top2_label, r.top3_label])
                for r, accepted in zip(group.itertuples(), group.accepted_labels)
            ])
            lo, hi = exact_ci(int(correct.sum()), correct.size)
            t3_lo, t3_hi = exact_ci(int(top3.sum()), top3.size)
            conf_lo, conf_hi = bootstrap_ci(group.confidence.to_numpy(), rng, n_boot)
            ci_rows.append({
                "cohort": cohort,
                "panel_size": panel,
                "n_samples": correct.size,
                "accepted_top1": correct.mean(),
                "accepted_top1_95ci_low": lo,
                "accepted_top1_95ci_high": hi,
                "accepted_top3": top3.mean(),
                "accepted_top3_95ci_low": t3_lo,
                "accepted_top3_95ci_high": t3_hi,
                "mean_confidence": group.confidence.mean(),
                "mean_confidence_bootstrap_95ci_low": conf_lo,
                "mean_confidence_bootstrap_95ci_high": conf_hi,
            })

        wide = pred.pivot(index="external_index", columns="panel_size", values="accepted_correct")
        for small, large in zip(PANEL_ORDER[:-1], PANEL_ORDER[1:]):
            a = wide[small].astype(bool)
            b = wide[large].astype(bool)
            small_only = int((a & ~b).sum())
            large_only = int((~a & b).sum())
            discordant = small_only + large_only
            p = 1.0 if discordant == 0 else binomtest(min(small_only, large_only), discordant, 0.5).pvalue
            comparison_rows.append({
                "cohort": cohort,
                "comparison": f"{large} vs {small}",
                "small_only_correct": small_only,
                "large_only_correct": large_only,
                "discordant_pairs": discordant,
                "exact_mcnemar_p_value": p,
            })

    pd.DataFrame(ci_rows).to_csv(outdir / "external_validation_confidence_intervals.tsv", sep="\t", index=False)
    pd.DataFrame(comparison_rows).to_csv(outdir / "external_validation_paired_mcnemar.tsv", sep="\t", index=False)


def write_summary(outdir: Path) -> None:
    internal = pd.read_csv(outdir / "internal_cv_bootstrap_ci.tsv", sep="\t")
    paired = pd.read_csv(outdir / "internal_cv_paired_panel_tests.tsv", sep="\t")
    external = pd.read_csv(outdir / "external_validation_confidence_intervals.tsv", sep="\t")
    mcnemar = pd.read_csv(outdir / "external_validation_paired_mcnemar.tsv", sep="\t")
    def markdown_table(df: pd.DataFrame) -> str:
        formatted = df.copy()
        for column in formatted.select_dtypes(include=["float"]).columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.4g}")
        headers = [str(column) for column in formatted.columns]
        body = [[str(value) for value in row] for row in formatted.itertuples(index=False, name=None)]
        return "\n".join([
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in body],
        ])

    lines = [
        "# Manuscript statistical analysis",
        "",
        "Internal confidence intervals use 20,000 bootstrap resamples of the 15 repeated-CV fold estimates. "
        "Adjacent panel sizes are compared on matched repeat-fold units using two-sided Wilcoxon signed-rank tests; "
        "Holm correction is applied within each metric.",
        "",
        "External top-1 and top-3 intervals are exact 95% binomial confidence intervals. Adjacent panel sizes are "
        "compared on matched samples using an exact McNemar test. Confidence intervals for mean model confidence "
        "use 20,000 sample-level bootstrap resamples.",
        "",
        "## Internal CV macro F1",
        "",
        markdown_table(internal[internal.metric == "macro_f1"]),
        "",
        "## Paired internal comparisons",
        "",
        markdown_table(paired[paired.metric == "macro_f1"]),
        "",
        "## Formal external validation",
        "",
        markdown_table(external),
        "",
        "## External matched-panel comparisons",
        "",
        markdown_table(mcnemar),
        "",
    ]
    (outdir / "manuscript_statistical_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    outdir = ROOT / "results/manuscript_statistics"
    outdir.mkdir(parents=True, exist_ok=True)
    internal_statistics(outdir)
    external_statistics(outdir, args.bootstrap)
    write_summary(outdir)


if __name__ == "__main__":
    main()
