#!/usr/bin/env python3
"""Build submission-facing LaTeX tables directly from audited result files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission/manuscript/manuscript_tables.tex"
SUPP_OUT = ROOT / "submission/manuscript/supplementary_tables.tex"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def bootstrap_ci(row: pd.Series) -> str:
    return f"{fmt(row.bootstrap_95ci_low)}--{fmt(row.bootstrap_95ci_high)}"


def exact_ci(row: pd.Series) -> str:
    return f"{fmt(row.accepted_top1_95ci_low)}--{fmt(row.accepted_top1_95ci_high)}"


def main() -> None:
    internal = pd.read_csv(ROOT / "results/manuscript_statistics/internal_cv_bootstrap_ci.tsv", sep="\t")
    paired = pd.read_csv(ROOT / "results/manuscript_statistics/internal_cv_paired_panel_tests.tsv", sep="\t")
    external = pd.read_csv(ROOT / "results/manuscript_statistics/external_validation_confidence_intervals.tsv", sep="\t")

    cohort_rows = [
        ("TCGA candidate", "Source audit", "9,812", "33 classes; 747 normal", "Normal rows excluded from formal training", "samples-used.tsv"),
        ("TCGA formal", "Development CV", "9,065", "33 tumour classes", "Patient-level 5-fold x 3 CV", "samples-locked.tsv"),
        ("GSE56044", "External", "106", "LUAD/LUSC", "Exact histology", "Selected cohort"),
        ("GSE48684", "External", "64", "COAD/READ", "Accepted colorectal lineage", "Selected cohort"),
        ("GSE53051", "External", "102", "Five prespecified lineages", "Accepted lineage", "Selected cohort"),
        ("GSE105260", "External", "35", "KIRC", "Exact histology", "Official beta"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Cohort and data summary. The 9,812-row candidate manifest contained 747 normal samples; only the 9,065 tumour rows in the locked manifest were used for formal model development and external-model training.}",
        r"\label{tab:cohort}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}p{2.0cm}p{2.5cm}rp{3.0cm}p{4.0cm}p{2.4cm}@{}}",
        r"\toprule",
        r"Cohort & Role & $n$ & Classes/labels & Evaluation definition & Provenance \\",
        r"\midrule",
    ]
    lines += ["{} & {} & {} & {} & {} & {} ".format(*row) + r"\\" for row in cohort_rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]

    lines += [
        r"\begin{table*}[t]",
        r"\caption{Internal repeated cross-validation performance for the principal panels. Values are means over 15 held-out evaluations; SD and fold-resampling 95\% CIs are reported. The change column is the mean macro-F1 difference from the preceding panel. Holm-adjusted $P$ values compare adjacent panel sizes for macro-F1 within the internal repeated-fold analysis.}",
        r"\label{tab:internal}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}rrrrrrrrrl@{}}",
        r"\toprule",
        r"Panel & Macro-F1 mean & SD & 95\% CI & Accuracy & Balanced accuracy & Top-3 & $\Delta$ Macro-F1 & Folds & Holm-adjusted $P$ \\",
        r"\midrule",
    ]
    for panel in [100, 200, 500, 1000]:
        mf = internal[(internal.panel_size == panel) & (internal.metric == "macro_f1")].iloc[0]
        acc = internal[(internal.panel_size == panel) & (internal.metric == "accuracy")].iloc[0]
        bal = internal[(internal.panel_size == panel) & (internal.metric == "balanced_accuracy")].iloc[0]
        top3 = internal[(internal.panel_size == panel) & (internal.metric == "top3_accuracy")].iloc[0]
        if panel == 100:
            p = r"\textemdash"
            delta = r"\textemdash"
        else:
            previous = {200: 100, 500: 200, 1000: 500}[panel]
            p_row = paired[(paired.comparison == f"{panel} vs {previous}") & (paired.metric == "macro_f1")].iloc[0]
            p = f"{p_row.holm_adjusted_p_value:.3g}"
            previous_mf = internal[(internal.panel_size == previous) & (internal.metric == "macro_f1")].iloc[0]
            delta = f"{mf['mean'] - previous_mf['mean']:.3f}"
        row = f"{panel:,} & {fmt(mf['mean'])} & {fmt(mf['sd'])} & {bootstrap_ci(mf)} & {fmt(acc['mean'])} & {fmt(bal['mean'])} & {fmt(top3['mean'])} & {delta} & {int(mf['n_folds'])} & {p} "
        lines.append(row + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]

    anchor = external[external.panel_size.eq(1000)].copy()
    lines += [
        r"\begin{table*}[t]",
        r"\caption{Formal external validation at the 1,000-CpG anchor panel. Each row reports one evaluation endpoint: exact accuracy when the external label is directly comparable to a TCGA class, or accepted-lineage accuracy when the prespecified lineage mapping was required. Values are shown as correct/$n$ and exact two-sided binomial 95\% CIs.}",
        r"\label{tab:external}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrlrlrl@{}}",
        r"\toprule",
        r"Cohort & $n$ & Endpoint & Correct/$n$ & Accuracy & 95\% CI & Definition \\",
        r"\midrule",
    ]
    definitions = {
        "GSE56044": ("Exact", "LUAD/LUSC"),
        "GSE48684": ("Accepted lineage", "COAD/READ"),
        "GSE53051": ("Accepted lineage", "Five-lineage mapping"),
        "GSE105260_official_series": ("Exact", "KIRC"),
    }
    for _, row in anchor.iterrows():
        cohort = row.cohort
        n = int(row.n_samples)
        accepted_correct = int(round(row.accepted_top1 * n))
        label = "GSE105260" if cohort == "GSE105260_official_series" else cohort
        endpoint, definition = definitions[cohort]
        latex_row = f"{label} & {n} & {endpoint} & {accepted_correct}/{n} & {fmt(row.accepted_top1)} & {exact_ci(row)} & {definition} "
        lines.append(latex_row + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]

    main_lines = list(lines)
    lines = []
    exploratory = [
        ("GSE56044", "1,000", "103/106", "Exact LUAD/LUSC"),
        ("GSE48684", "200 and 500 (tie)", "57/64", "Accepted COAD/READ lineage"),
        ("GSE53051", "1,000", "92/102", "Accepted five-lineage mapping"),
        ("GSE105260", "1,000", "31/35", "Exact KIRC"),
    ]
    lines += [
        r"\begin{table*}[t]",
        r"\caption{Exploratory best panel per external cohort. These choices were made after observing the external results and were not used for prespecified model selection or refitting.}",
        r"\label{tab:exploratory-panel}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lrlp{5.0cm}@{}}",
        r"\toprule",
        r"Cohort & Exploratory panel & Correct/$n$ & Evaluation definition \\",
        r"\midrule",
    ]
    lines += [f"{a} & {b} & {c} & {d} " + r"\\" for a, b, c, d in exploratory]
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    comparator = [
        ("Logistic regression", "Variance, 500 CpGs", "0.920", "0.924", "0.944", "0.993", "15"),
        ("Logistic regression", "Variance, 1,000 CpGs", "0.931", "0.934", "0.953", "0.994", "15"),
        ("MLP", "Variance, 500 CpGs", "0.911", "0.907", "0.938", "0.991", "15"),
        ("MLP", "Variance, 1,000 CpGs", "0.904", "0.905", "0.940", "0.991", "15"),
        ("Random forest", "Variance, 1,000 CpGs", "0.858", "0.855", "0.919", "0.986", "15"),
        ("SGD log-loss", "Variance, 1,000 CpGs", "0.912", "0.906", "0.946", "0.965", "15"),
    ]
    lines += [
        r"\begin{table*}[t]",
        r"\caption{Internal comparator analysis on the locked 15-fold evaluation units. Values are means over the completed 15-fold runs. This ancillary comparison was used to contextualise the primary logistic-regression analysis and was not used to select the external anchor panel.}",
        r"\label{tab:baseline-comparison}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Model & Feature/panel & Macro-F1 & Balanced accuracy & Accuracy & Top-3 accuracy & Completed folds \\",
        r"\midrule",
    ]
    lines += ["{} & {} & {} & {} & {} & {} & {} ".format(*row) + r"\\" for row in comparator]
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""]
    OUT.write_text("\n".join(main_lines), encoding="utf-8")
    SUPP_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
