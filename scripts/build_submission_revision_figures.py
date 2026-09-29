"""Build submission figures from locked results without model retraining.

The script deliberately separates exploratory all-panel screening summaries from
the frozen patient-level and external-validation estimands used in the main text.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from _figure_style import (  # noqa: E402
    BLACK, GREY, LIGHT, BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT, RED,
    PALE_BLUE, PALE_ORANGE, BLUE_TINT, ORANGE_TINT, GREEN_TINT,
    CMAP_SEQ, CMAP_DIV,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "manuscript_figures_frozen_revision"
OUT.mkdir(exist_ok=True, parents=True)
FROZEN = ROOT / "server_results_20260728" / "results"
LOCKED = FROZEN / "internal_cv" / "lr_cpg_panel_frozen_20260721"
STATS = FROZEN / "manuscript_statistics_frozen_20260721"
ASSETS = FROZEN / "manuscript_assets_frozen_20260721" / "tables"
ROBUST = FROZEN / "robustness" / "frozen_20260721"
CURRENT = ROOT / "results" / "internal_cv" / "lr_cpg_panel"

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 8.2,
                     "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "axes.linewidth": .75,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     # Raster elements embedded in the PDF (imshow panels, colour
                     # bars, dense scatter) must meet the >=300 dpi floor journals
                     # require; the matplotlib default is only 100 dpi.
                     "figure.dpi": 600, "savefig.dpi": 600})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def label(ax, value):
    ax.text(-.12, 1.05, str(value).upper(), transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="bottom")


def fig1():
    fig, ax = plt.subplots(figsize=(7.15, 2.25)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nodes = [(0.02, "Development cohort", "9,065 tumours\n8,923 patients\n33 classes"),
             (0.27, "Patient-level CV", "5 folds × 3 repeats\npreprocessing fitted\nwithin each fold"),
             (0.52, "Consensus panels", "15 training-fold selections\nlocked 500-CpG and\n1,000-CpG panels"),
             (0.77, "External evaluation", "4 GEO cohorts\nn = 307\nendpoint-specific evaluation")]
    for x, title, body in nodes:
        ax.add_patch(Rectangle((x, .30), .20, .38, facecolor="white", edgecolor=BLACK, linewidth=.75))
        ax.text(x+.10, .61, title, ha="center", va="center", fontweight="bold", fontsize=6.6)
        ax.text(x+.10, .44, body, ha="center", va="center", fontsize=5.6, linespacing=1.20)
    for x in (.22, .47, .72): ax.annotate("", xy=(x+.045, .49), xytext=(x, .49), arrowprops={"arrowstyle": "-|>", "lw": .7, "color": GREY})
    ax.text(.02, .88, "Leakage-free development and locked external evaluation", fontweight="bold", fontsize=8.5)
    ax.text(.02, .10, "Primary internal estimand: repeat-specific patient-level out-of-fold metrics; external models fitted once on all development tumours.", fontsize=6.1, color=GREY)
    save(fig, "figure1_study_design")


def fig2():
    screen = pd.read_csv(CURRENT / "panel_performance_pretty.csv")
    folds = pd.read_csv(CURRENT / "fold_metrics.tsv", sep="\t")
    primary = pd.read_csv(STATS / "internal_patient_oof_primary_estimates_ci.tsv", sep="\t")
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    for col, color, name in [("macro_f1_mean", BLUE, "Macro-F1"), ("accuracy_mean", ORANGE, "Accuracy"), ("top3_accuracy_mean", RED, "Top-3 accuracy")]:
        ax.plot(screen.panel_size, screen[col], "o-", color=color, lw=1, ms=3, label=name)
    ax.set_xscale("log"); ax.set_xticks(screen.panel_size, [str(x) for x in screen.panel_size], rotation=45)
    ax.set_ylim(0, 1.03); ax.set_xlabel("CpG panel size"); ax.set_ylabel("Mean held-out fold metric")
    ax.legend(frameon=False, fontsize=6, loc="lower right"); clean(ax); label(ax, "a")
    ax.set_title("All-panel screening series", loc="left", fontweight="bold")
    ax = axes[1]
    for i, panel in enumerate((500, 1000)):
        sub = primary[(primary.panel_size == panel) & primary.metric.isin(["macro_f1", "accuracy"])].set_index("metric")
        for j, metric in enumerate(("macro_f1", "accuracy")):
            r = sub.loc[metric]; x = j + (-.12 if panel == 500 else .12)
            ax.errorbar(x, r.estimate_mean_across_repeats, yerr=[[r.estimate_mean_across_repeats-r.ci_low], [r.ci_high-r.estimate_mean_across_repeats]], fmt="o", color=BLUE if panel == 500 else ORANGE, capsize=2, ms=4, label=f"{panel:,} CpG" if j == 0 else None)
    ax.set_xticks([0,1], ["Macro-F1", "Accuracy"]); ax.set_ylim(.90,.97); ax.set_ylabel("Patient-level estimate (95% CI)")
    ax.legend(frameon=False, fontsize=6, loc="lower right"); clean(ax); label(ax, "b"); ax.set_title("Primary patient-level estimand", loc="left", fontweight="bold")
    ax = axes[2]
    positions = np.arange(len(screen)); data = [folds.loc[folds.panel_size == p, "macro_f1"].to_numpy() for p in screen.panel_size]
    ax.boxplot(data, positions=positions, widths=.55, showfliers=False, patch_artist=True, boxprops={"facecolor":"white","edgecolor":BLACK,"linewidth":.7}, medianprops={"color":ORANGE,"linewidth":1}, whiskerprops={"color":BLACK,"linewidth":.7}, capprops={"color":BLACK,"linewidth":.7})
    ax.set_xticks(positions, [str(x) for x in screen.panel_size], rotation=45); ax.set_ylim(0,1.02); ax.set_xlabel("CpG panel size"); ax.set_ylabel("Fold macro-F1")
    clean(ax); label(ax, "c"); ax.set_title("Variation across 15 held-out folds", loc="left", fontweight="bold")
    fig.subplots_adjust(wspace=.48); save(fig, "figure2_internal_stability")


def fig3():
    frozen = pd.read_csv(ASSETS / "table_external_frozen_endpoints_ci.tsv", sep="\t")
    screen = pd.read_csv(ROOT / "results" / "manuscript_statistics" / "external_validation_confidence_intervals.tsv", sep="\t")
    cohorts = ["GSE56044", "GSE53051", "GSE48684", "GSE105260"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.5), gridspec_kw={"width_ratios":[1.0,1.2,1.0]})
    ax = axes[0]; mat = screen.pivot(index="cohort", columns="panel_size", values="accepted_top1").reindex(["GSE56044","GSE53051","GSE48684","GSE105260_official_series"])
    im = ax.imshow(mat, vmin=.60, vmax=1, cmap=CMAP_SEQ, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,f"{mat.iloc[i,j]:.2f}",ha="center",va="center",fontsize=6,color="white" if mat.iloc[i,j]>.83 else BLACK)
    ax.set_xticks(range(4), ["100","200","500","1,000"]); ax.set_yticks(range(4), ["GSE56044\n33-class","GSE53051\n6-state","GSE48684\npositive-only","GSE105260\npositive-only"]); ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlabel("CpG panel size"); ax.set_title("Screening endpoint estimates",loc="left",fontweight="bold"); label(ax,"a")
    ax = axes[1]; rows = frozen.sort_values(["cohort","panel_size"], ascending=[True, False]).reset_index(drop=True)
    for i, r in rows.iterrows():
        color = BLUE if r.panel_size == 500 else ORANGE
        ax.errorbar(r.estimate, i, xerr=[[r.estimate-r.exact_binomial_95ci_low],[r.exact_binomial_95ci_high-r.estimate]], fmt="o", color=color, capsize=2, ms=3.5)
        ax.text(r.exact_binomial_95ci_low-.008, i, f"{r.successes}/{r.n_samples}", ha="right", va="center", fontsize=5.8)
    ax.set_yticks(range(len(rows)), [f"{r.cohort} — {int(r.panel_size):,}" for _,r in rows.iterrows()]); ax.tick_params(axis="y",labelsize=5.2); ax.set_xlim(.60,1.01); ax.set_xlabel("Formal endpoint estimate (95% CI)")
    clean(ax); ax.grid(axis="x",color=LIGHT,lw=.7); ax.set_title("Frozen external estimands",loc="left",fontweight="bold"); label(ax,"b")
    ax = axes[2]
    for cohort, color in [("GSE56044", BLUE), ("GSE53051", ORANGE)]:
        pred = pd.read_csv(ROOT / "results" / "external_validation" / cohort / "lr_panel" / "external_predictions.tsv", sep="\t"); pred = pred[pred.panel_size == 1000]
        thresholds = np.arange(.5, 1.0, .05); coverage = [(pred.confidence >= t).mean() for t in thresholds]
        ax.plot(thresholds, coverage, "o-", color=color, ms=2.5, lw=1, label=cohort)
    ax.set_ylim(0,1.02); ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Retained sample fraction"); ax.legend(frameon=False,fontsize=6)
    clean(ax); ax.set_title("Sample-level confidence coverage",loc="left",fontweight="bold"); label(ax,"c")
    fig.subplots_adjust(wspace=.95); save(fig,"figure3_external_validation")


def fig4():
    stable = pd.read_csv(CURRENT / "stable_cpg_panel_annotated.tsv", sep="\t")
    cls = pd.read_csv(CURRENT / "class_specific_top_cpg_annotated.tsv", sep="\t")
    full = stable[stable.panel_size == 1000].copy()
    top = full.sort_values(["selected_count","mean_rank"], ascending=[False,True]).head(100)
    fig, axes = plt.subplots(2,2,figsize=(7.15,4.65)); ax=axes[0,0]
    counts = full.selected_count.value_counts().sort_index(); ax.bar(counts.index, counts.values, color=GREY); ax.set_xlabel("Selections across 15 training folds"); ax.set_ylabel("CpGs in the 1,000-CpG union"); clean(ax); label(ax,"a"); ax.set_title("Recurrence of selected CpGs",loc="left",fontweight="bold")
    ax=axes[0,1]; context=pd.Series({"Promoter":top.is_promoter.fillna(False).sum(),"Island":top.relation_to_cpg_island.fillna("").eq("Island").sum(),"Shore":top.relation_to_cpg_island.fillna("").str.contains("Shore").sum(),"Gene annotated":top.gene_symbol.fillna("").ne("").sum()}); ax.barh(context.index,context.values,color=BLUE); ax.set_xlabel("CpGs among stable top 100"); clean(ax); label(ax,"b"); ax.set_title("Genomic annotation context",loc="left",fontweight="bold")
    ax=axes[1,0]; genes=cls[cls.gene_symbol.fillna("").ne("")].groupby("gene_symbol").mean_abs_coef.mean().nlargest(12).sort_values(); ax.barh(genes.index,genes.values,color=BLACK); ax.set_xlabel("Mean absolute class coefficient"); clean(ax); label(ax,"c"); ax.set_title("Class-associated annotated genes",loc="left",fontweight="bold")
    ax=axes[1,1]; selected=list(genes.index[-10:]); classes=["TCGA-BRCA","TCGA-COAD","TCGA-READ","TCGA-KIRC","TCGA-KIRP","TCGA-LUAD","TCGA-LUSC","TCGA-PAAD","TCGA-THCA"]; heat=cls[cls.gene_symbol.isin(selected)&cls.label.isin(classes)].pivot_table(index="gene_symbol",columns="label",values="mean_coef",aggfunc="mean",fill_value=0).reindex(index=selected,columns=classes).fillna(0); vmax=np.quantile(np.abs(heat.to_numpy()),.95) or 1; im=ax.imshow(heat,cmap=CMAP_DIV,vmin=-vmax,vmax=vmax,aspect="auto"); ax.set_xticks(range(len(classes)),[x.replace("TCGA-","") for x in classes],rotation=45,ha="right"); ax.set_yticks(range(len(selected)),selected); ax.set_title("Class-specific coefficient pattern",loc="left",fontweight="bold"); label(ax,"d"); fig.colorbar(im,ax=ax,fraction=.05,pad=.02)
    fig.text(.01,.01,"Annotation and coefficient summaries are descriptive; they do not establish biological mechanism or pathway enrichment.",fontsize=6,color=GREY)
    fig.subplots_adjust(hspace=.55,wspace=.55,bottom=.10); save(fig,"figure4_annotation_ood")


def error_metrics():
    raw=pd.read_csv(LOCKED / "repeated_cross_fitting_ensemble_confusion_matrices.tsv",sep="\t"); labels=[c for c in raw.columns if c.startswith("TCGA-")]; mat=raw[raw.panel_size==1000].set_index("true_label").loc[labels,labels].to_numpy(float); norm=mat/mat.sum(1,keepdims=True); return labels,mat,norm


def fig5_and_supp():
    labels, mat, norm = error_metrics(); short=[x.replace("TCGA-","") for x in labels]
    pairs=[]
    for i in range(len(labels)):
        row=norm[i].copy(); row[i]=0; j=int(row.argmax()); pairs.append((row[j], short[i], short[j], int(mat[i,j])))
    pairs=sorted(pairs,reverse=True)[:8]
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.55),gridspec_kw={"width_ratios":[1.25,1]}); ax=axes[0]
    y=np.arange(len(pairs))[::-1]; ax.barh(y,[p[0] for p in pairs],color=GREY); ax.set_yticks(y,[f"{p[1]} → {p[2]} (n={p[3]})" for p in pairs]); ax.set_xlim(0,max(p[0] for p in pairs)*1.2); ax.set_xlabel("Within-class error proportion, 1,000 CpGs"); clean(ax); label(ax,"a"); ax.set_title("Largest directional class confusions",loc="left",fontweight="bold")
    effects=[]
    for p in (500,1000):
        e=pd.read_csv(ROBUST / f"consensus_locked_{p}_cpg" / "robustness_confirmatory_effects_bootstrap_ci.tsv",sep="\t"); e["panel"]=p; effects.append(e)
    e=pd.concat(effects); ax=axes[1]; order=["added_normal_profile_fraction","uniform_panel_probe_loss"]
    for p,color,off in [(500,BLUE,-.17),(1000,ORANGE,.17)]:
        sub=e[e.panel==p].set_index("condition").loc[order]; vals=sub.estimate_delta.to_numpy(); err=np.vstack([vals-sub.ci_low.to_numpy(),sub.ci_high.to_numpy()-vals]); ax.bar(np.arange(2)+off,vals,.3,color=color,label=f"{p:,} CpG"); ax.errorbar(np.arange(2)+off,vals,yerr=err,fmt="none",ecolor=BLACK,capsize=2,lw=.7)
    ax.axhline(0,color=GREY,lw=.7); ax.set_xticks([0,1],["40% added\nnormal profile","20% uniform\nprobe loss"]); ax.set_ylabel(r"$\Delta$ macro-F1 vs clean input"); ax.legend(frameon=False,fontsize=6); clean(ax); label(ax,"b"); ax.set_title("Fixed-model input sensitivity",loc="left",fontweight="bold")
    fig.subplots_adjust(wspace=.5); save(fig,"figure5_class_performance")
    fig,axes=plt.subplots(1,2,figsize=(7.15,5.3),gridspec_kw={"width_ratios":[1.1,.9]}); ax=axes[0]; im=ax.imshow(norm,vmin=0,vmax=1,cmap=CMAP_SEQ,aspect="auto"); ax.set_xticks(range(len(short)),short,rotation=90); ax.set_yticks(range(len(short)),short); ax.set_xlabel("Predicted class"); ax.set_ylabel("Reference class"); ax.set_title("Full 33-class confusion matrix",loc="left",fontweight="bold"); fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
    ax=axes[1]; f1=[]
    for i in range(len(labels)):
        tp=mat[i,i]; recall=tp/mat[i].sum(); precision=tp/mat[:,i].sum(); f1.append(2*precision*recall/(precision+recall))
    order=np.argsort(f1); ax.scatter(np.array(f1)[order],np.arange(len(order)),color=BLUE,s=14); ax.set_yticks(np.arange(len(order)),[short[i] for i in order]); ax.set_xlabel("1,000-CpG descriptive class F1"); ax.grid(axis="x",color=LIGHT,lw=.7); ax.spines[["top","right"]].set_visible(False); ax.set_title("Full class-level summary",loc="left",fontweight="bold")
    fig.subplots_adjust(wspace=.45); save(fig,"supp_figure3_full_class_performance")


def supp_robustness_series():
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    for panel, color in ((500, BLUE), (1000, ORANGE)):
        summary = pd.read_csv(ROBUST / f"consensus_locked_{panel}_cpg" / "robustness_seed_summary.tsv", sep="\t")
        sub = summary[(summary.condition == "added_normal_profile_fraction") & (summary.metric == "macro_f1")].sort_values("level")
        x = sub["level"].to_numpy(float); median = sub["seed_median"].to_numpy(float)
        ax.fill_between(x, sub.seed_q_0_025.to_numpy(float), sub.seed_q_0_975.to_numpy(float), color=color, alpha=.13, linewidth=0)
        ax.plot(x, median, "o-", color=color, ms=3.5, lw=1.1, label=f"{panel:,} CpG")
    ax.set_xticks([0,.2,.4,.6,.8]); ax.set_ylim(.58,1.02); ax.set_xlabel("Added same-project normal-profile fraction"); ax.set_ylabel("Seed-median macro-F1"); ax.legend(frameon=False, fontsize=6, loc="center left")
    clean(ax); ax.set_title("Full normal-admixture sensitivity series", loc="left", fontweight="bold")
    # Two provenance notes that the plot alone cannot carry: the unperturbed point
    # is an in-sample reference, and the lambda = 0.2 level contains a small group of
    # degenerately unperturbed 1,000-CpG seeds, which inflates its upper percentile
    # band.  Both notes are descriptive; no confirmatory contrast uses lambda = 0.2.
    ax.text(.98, .965,
            "$\\lambda$ = 0: in-sample reference (models fitted on all 9,065 tumours)\n"
            "$\\lambda$ = 0.2: 6/50 seeds of the 1,000-CpG panel reproduced unperturbed predictions;\n"
            "this level is descriptive only and is not used for any confirmatory contrast",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.0, color=GREY,
            linespacing=1.4,
            bbox={"facecolor":"white","edgecolor":LIGHT,"linewidth":.5,"pad":2.2,"alpha":.95})
    save(fig,"supp_figure4_robustness_series")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5_and_supp(); supp_robustness_series()
    print(f"Wrote revision figures to {OUT}")

if __name__ == "__main__": main()
