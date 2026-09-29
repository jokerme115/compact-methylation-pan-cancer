"""Build Supplementary Data 1 from frozen manuscript result tables.

This script reads existing derived results only; it does not train or refit models.
"""
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "teacher_revision_20260815"
XLSX = OUT / "Supplementary_Data_1.xlsx"

SOURCES = {
    "Panel_screen": (ROOT / "results/internal_cv/lr_cpg_panel/panel_performance_pretty.csv", ","),
    "Primary_OOF": (ROOT / "server_results_20260728/results/manuscript_statistics_frozen_20260721/internal_patient_oof_primary_estimates_ci.tsv", "\t"),
    "External": (ROOT / "server_results_20260728/results/manuscript_assets_frozen_20260721/tables/table_external_frozen_endpoints_ci.tsv", "\t"),
    "Stability": (ROOT / "server_results_20260728/results/manuscript_assets_frozen_20260721/tables/table_panel_stability.tsv", "\t"),
    "Class_metrics": (ROOT / "results/manuscript_figures_frozen_revision/internal_class_level_metrics.tsv", "\t"),
    "Normal_tissue": (ROOT / "server_results_20260728/results/manuscript_assets_frozen_20260721/tables/table_normal_ood_summary.tsv", "\t"),
    "Locked_panels": (ROOT / "server_results_20260728/results/internal_cv/lr_cpg_panel_frozen_20260721/consensus_locked_panels.tsv", "\t"),
}


def load_tables():
    tables = {name: pd.read_csv(path, sep=sep) for name, (path, sep) in SOURCES.items()}
    panel = tables.pop("Locked_panels")
    tables["Panel_500"] = panel.loc[panel["panel_size"].eq(500)].reset_index(drop=True)
    tables["Panel_1000"] = panel.loc[panel["panel_size"].eq(1000)].reset_index(drop=True)

    admixture = []
    probe_loss = []
    for size in (500, 1000):
        a = pd.read_csv(
            ROOT / f"server_results_20260728/results/robustness/frozen_20260721/consensus_locked_{size}_cpg/robustness_confirmatory_effects_bootstrap_ci.tsv",
            sep="\t",
        )
        a = a.loc[a["condition"].eq("added_normal_profile_fraction")].copy()
        a.insert(0, "panel_size", size)
        a.insert(1, "analysis", "in_silico_tissue_matched_normal_admixture")
        admixture.append(a)
        p = pd.read_csv(
            ROOT / f"results/robustness/oof_probe_loss_frozen_20260728/consensus_locked_{size}_cpg/oof_probe_loss_confirmatory_patient_bootstrap_ci.tsv",
            sep="\t",
        )
        p.insert(1, "analysis", "separately_refitted_patient_level_oof_probe_loss")
        probe_loss.append(p)
    tables["Robustness"] = pd.concat(admixture + probe_loss, ignore_index=True, sort=False)
    return tables


def write_workbook(tables):
    OUT.mkdir(parents=True, exist_ok=True)
    title = "Compact DNA methylation panels for pan-cancer lineage classification"
    boundaries = {
        "Panel_screen": "Fold-level screening; not the primary patient-level estimate.",
        "Primary_OOF": "Patient-level OOF bootstrap; patient resampled once while retaining repeat predictions.",
        "External": "Endpoint-specific results; positive-only cohorts estimate recall, not specificity.",
        "Stability": "Selection reproducibility; not evidence of biological validity.",
        "Class_metrics": "Descriptive repeat-averaged OOF error localisation; not the primary estimate.",
        "Normal_tissue": "TCGA normal-tissue stress test; not a general clinical OOD cohort.",
        "Robustness": "Normal admixture and OOF probe loss are distinct analyses with distinct baselines.",
        "Panel_500": "Locked compact candidate panel, ordered by consensus rank.",
        "Panel_1000": "Locked higher-dimensional reference panel, ordered by consensus rank.",
    }
    source_rows = []
    for sheet in tables:
        if sheet.startswith("Panel_"):
            path = SOURCES["Locked_panels"][0]
        elif sheet == "Robustness":
            path = "Two frozen result families; see worksheet boundary note"
        else:
            path = SOURCES[sheet][0]
        source_rows.append([sheet, len(tables[sheet]), str(path), boundaries[sheet]])
    dictionary = pd.DataFrame(source_rows, columns=["worksheet", "data_rows", "source", "reporting_boundary"])

    with pd.ExcelWriter(XLSX, engine="xlsxwriter") as writer:
        wb = writer.book
        title_fmt = wb.add_format({"bold": True, "font_size": 15, "font_color": "white", "bg_color": "#245B84", "align": "left", "valign": "vcenter"})
        note_fmt = wb.add_format({"font_color": "#404040", "bg_color": "#EAF2F8", "text_wrap": True, "valign": "top"})
        head_fmt = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#3677A5", "border": 1, "text_wrap": True, "valign": "top"})
        body_fmt = wb.add_format({"border": 1, "border_color": "#D9E1E8", "valign": "top"})
        num_fmt = wb.add_format({"border": 1, "border_color": "#D9E1E8", "num_format": "0.0000"})
        int_fmt = wb.add_format({"border": 1, "border_color": "#D9E1E8", "num_format": "0"})

        sheets = [("README", dictionary)] + list(tables.items())
        for name, df in sheets:
            df.to_excel(writer, sheet_name=name, startrow=4, index=False, header=False)
            ws = writer.sheets[name]
            ws.set_row(0, 25)
            ws.merge_range(0, 0, 0, max(3, len(df.columns)-1), title if name == "README" else f"Supplementary Data 1 — {name}", title_fmt)
            note = ("Workbook data dictionary and provenance. Derived results only; no restricted raw methylation values."
                    if name == "README" else boundaries[name])
            ws.merge_range(1, 0, 2, max(3, len(df.columns)-1), note, note_fmt)
            for c, col in enumerate(df.columns):
                ws.write(3, c, col, head_fmt)
                series = df[col]
                if pd.api.types.is_integer_dtype(series):
                    ws.set_column(c, c, min(max(len(str(col))+2, 12), 24), int_fmt)
                elif pd.api.types.is_numeric_dtype(series):
                    ws.set_column(c, c, min(max(len(str(col))+2, 14), 24), num_fmt)
                else:
                    max_len = max([len(str(col))] + [len(str(x)) for x in series.head(200).fillna("")])
                    ws.set_column(c, c, min(max(max_len + 2, 12), 46), body_fmt)
            ws.freeze_panes(4, 0)
            ws.autofilter(3, 0, 3 + len(df), len(df.columns)-1)
            ws.set_landscape()
            ws.fit_to_pages(1, 0)
            ws.set_margins(0.3, 0.3, 0.5, 0.5)
            if len(df):
                ws.conditional_format(4, 0, 3 + len(df), len(df.columns)-1,
                                      {"type": "formula", "criteria": "=MOD(ROW(),2)=0",
                                       "format": wb.add_format({"bg_color": "#F7FAFC"})})

        readme = writer.sheets["README"]
        readme.write(4 + len(dictionary) + 2, 0, "Panel roles", head_fmt)
        readme.write(4 + len(dictionary) + 3, 0, "500 CpGs", body_fmt)
        readme.write(4 + len(dictionary) + 3, 1, "Compact candidate; not a prospectively selected clinical cutoff.", body_fmt)
        readme.write(4 + len(dictionary) + 4, 0, "1,000 CpGs", body_fmt)
        readme.write(4 + len(dictionary) + 4, 1, "Higher-dimensional reference anchor; not proof of universal superiority.", body_fmt)


def validate_and_preview(tables):
    assert XLSX.exists() and XLSX.stat().st_size > 10_000
    with zipfile.ZipFile(XLSX) as zf:
        xml = b"".join(zf.read(n) for n in zf.namelist() if n.endswith(".xml"))
    errors = [token for token in (b"#REF!", b"#DIV/0!", b"#VALUE!", b"#NAME?", b"#N/A") if token in xml]
    if errors:
        raise RuntimeError(f"Formula/error tokens found: {errors}")
    assert len(tables["Panel_500"]) == 500
    assert len(tables["Panel_1000"]) == 1000
    assert set(tables["Primary_OOF"]["panel_size"]) == {500, 1000}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    p = tables["Panel_screen"]
    axes[0].plot(p["panel_size"], p["macro_f1_mean"], "o-", color="#245B84")
    axes[0].set_xscale("log"); axes[0].set_ylim(0, 1)
    axes[0].set_title("Panel_screen preview"); axes[0].set_xlabel("CpG panel size"); axes[0].set_ylabel("Mean fold-level macro-F1")
    e = tables["External"].sort_values(["cohort", "panel_size"])
    labels = [f"{r.cohort}\n{int(r.panel_size)}" for _, r in e.iterrows()]
    axes[1].bar(np.arange(len(e)), e["estimate"], color=["#D55E00" if x == 1000 else "#245B84" for x in e["panel_size"]])
    axes[1].set_xticks(np.arange(len(e)), labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylim(0.6, 1); axes[1].set_title("External worksheet preview"); axes[1].set_ylabel("Endpoint estimate")
    fig.tight_layout()
    fig.savefig(OUT / "Supplementary_Data_1_preview.png", dpi=180)
    plt.close(fig)


def main():
    tables = load_tables()
    write_workbook(tables)
    validate_and_preview(tables)
    print(f"Wrote {XLSX}")
    print("Sheets:", ", ".join(["README"] + list(tables)))


if __name__ == "__main__":
    main()
