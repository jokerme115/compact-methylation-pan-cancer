"""Build the final submission figures from existing locked result files only."""
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np
import pandas as pd

from build_submission_revision_figures import (
    ROOT, OUT, LOCKED, STATS, ASSETS, ROBUST, CURRENT,
    BLACK, GREY, LIGHT, BLUE, BLUE_LIGHT, ORANGE, ORANGE_LIGHT, RED,
    BLUE_TINT, ORANGE_TINT, PALE_ORANGE, CMAP_SEQ, CMAP_DIV,
    clean, label, save,
)

# Figure 3B carries four curves.  Each cohort-panel pair now draws its own hue
# from the shared palette (BLUE/BLUE_LIGHT for GSE56044, ORANGE/ORANGE_LIGHT for
# GSE53051); with only two hues the two 1,000-CpG curves were indistinguishable.
OOF_PROBE_LOSS = ROOT / "results" / "robustness" / "oof_probe_loss_frozen_20260728"
# Unified verified 16-size run: the same source as main Table 2 and Supplementary
# Table S13.  The earlier E1_panel_curve run covered only 10 panel sizes and its
# paired macro-F1 difference (0.0023) disagreed with the manuscript's 0.0026.
UNIFIED = ROOT / "outputs" / "panel_knee_v5_verified_audit_20260908" / "final_package" / "results"
LABELLED_PANELS = {500, 1000, 2000, 5000, 10000}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 8.3,
                     "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
                     "axes.linewidth": .75, "pdf.fonttype": 42, "ps.fonttype": 42})


def fig1():
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))

    # Two shaded zones distinguish model development from independent evaluation.
    ax.add_patch(FancyBboxPatch((.015,.35),.61,.58,boxstyle="round,pad=.012,rounding_size=.018",
                                facecolor=BLUE_TINT,edgecolor=BLUE,linewidth=.8))
    ax.add_patch(FancyBboxPatch((.64,.35),.345,.58,boxstyle="round,pad=.012,rounding_size=.018",
                                facecolor=ORANGE_TINT,edgecolor=ORANGE,linewidth=.8))
    ax.text(.035,.895,"DEVELOPMENT AND PANEL LOCKING",fontsize=6.8,fontweight="bold",color=BLUE)
    ax.text(.66,.895,"INDEPENDENT EVALUATION",fontsize=6.8,fontweight="bold",color=ORANGE)

    cards=[
        (.035,.57,.16,.24,"Development cohort","9,065 tumours\n8,923 patients\n33 TCGA classes",BLUE),
        (.225,.57,.17,.24,"Patient-level CV","5 folds × 3 repeats\nfold-specific processing\nand CpG ranking",BLUE),
        (.425,.57,.17,.24,"Locked panels","500 CpGs: compact\n1,000 CpGs: reference",ORANGE),
        (.665,.57,.135,.24,"Locked models","fit once on all\n9,065 tumours",ORANGE),
        (.83,.57,.13,.24,"Four GEO cohorts","307 samples\nno external tuning",ORANGE)]
    for i,(x,y,w,h,title,body,color) in enumerate(cards):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012,rounding_size=.014",
                                    facecolor="white",edgecolor=color,linewidth=1.0))
        ax.add_patch(Circle((x+.026,y+h-.043),.014,facecolor=color,edgecolor="none"))
        ax.text(x+.026,y+h-.043,str(i+1),ha="center",va="center",fontsize=5.4,
                color="white",fontweight="bold")
        ax.text(x+w/2,y+h-.085,title,ha="center",va="center",fontweight="bold",fontsize=6.0)
        ax.text(x+w/2,y+.075,body,ha="center",va="center",fontsize=5.45,linespacing=1.25)
    for x0,x1 in [(.195,.225),(.395,.425),(.595,.665),(.80,.83)]:
        ax.annotate("",xy=(x1-.004,.69),xytext=(x0+.004,.69),
                    arrowprops={"arrowstyle":"-|>","lw":1.0,"color":GREY})
    ax.plot([.625,.625],[.39,.91],color=GREY,lw=.9,ls=(0,(3,2)))
    ax.text(.625,.37,"panel and preprocessing parameters locked",ha="center",va="top",
            fontsize=5.2,color=GREY)

    ax.text(.025,.27,"PRIMARY EVALUATION AND ROBUSTNESS ANALYSES",fontsize=6.8,fontweight="bold",color=BLACK)
    branches=[
        (.03,.055,.21,"Performance","panel-size screen\npatient-level OOF and CIs",BLUE),
        (.275,.055,.21,"Reproducibility","selection recurrence\nclass-specific errors",BLUE),
        (.52,.055,.21,"Perturbation","normal admixture\n20% OOF probe loss",ORANGE),
        (.765,.055,.205,"Evaluation scope","747 normal tissues\nendpoint-aware reporting",ORANGE)]
    for x,y,w,title,body,color in branches:
        ax.add_patch(FancyBboxPatch((x,y),w,.145,boxstyle="round,pad=.009,rounding_size=.012",
                                    facecolor="white",edgecolor=LIGHT,linewidth=.7))
        ax.add_patch(FancyBboxPatch((x,y+.112),w,.033,boxstyle="round,pad=.002,rounding_size=.008",
                                    facecolor=color,edgecolor=color,linewidth=0))
        ax.text(x+w/2,y+.128,title,ha="center",va="center",fontsize=5.5,
                color="white",fontweight="bold")
        ax.text(x+w/2,y+.057,body,ha="center",va="center",fontsize=5.25,linespacing=1.18)
    save(fig,"figure1_study_design")


def fig2():
    curve=pd.read_csv(UNIFIED/"patient_oof_primary_estimates_ci.tsv",sep="\t")
    paired=pd.read_csv(UNIFIED/"paired_panel_comparison.tsv",sep="\t").iloc[0]
    primary=curve
    fig=plt.figure(figsize=(6.0,4.05))
    gs=fig.add_gridspec(2,2,width_ratios=[1.22,1],height_ratios=[1,1],
                        wspace=.45,hspace=.53)
    axes=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
    ax=axes[0]
    d=curve[curve.metric=="macro_f1"].sort_values("panel_size")
    x=np.arange(len(d)); y=d.estimate_mean_across_repeats.to_numpy()
    lo=y-d.ci_low.to_numpy(); hi=d.ci_high.to_numpy()-y
    boundary=np.flatnonzero(d.panel_size.to_numpy()==1000)[0]+.5
    ax.axvspan(boundary,len(d)-.5,color=PALE_ORANGE,zorder=0)
    ax.axvline(boundary,color=ORANGE_LIGHT,lw=.8,ls=(0,(3,2)),zorder=1)
    ax.errorbar(x,y,yerr=[lo,hi],fmt="o-",color=BLACK,lw=1.1,ms=3,
                ecolor=GREY,elinewidth=.8,capsize=1.8)
    for p,c in [(500,BLUE),(1000,ORANGE)]:
        i=np.flatnonzero(d.panel_size.to_numpy()==p)[0]
        ax.scatter(i,y[i],s=38,facecolor="white",edgecolor=c,linewidth=1.3,zorder=4)
    i1500=np.flatnonzero(d.panel_size.to_numpy()==1500)[0]
    ax.scatter(i1500,y[i1500],s=30,marker="D",facecolor=ORANGE,
               edgecolor="white",linewidth=.7,zorder=5)
    ax.annotate("1,500-CpG segmented candidate\n(not uniquely identified)",
                xy=(i1500,y[i1500]),xytext=(i1500-5.4,.9470),fontsize=7.0,color=RED,
                ha="left",va="top",arrowprops={"arrowstyle":"-","lw":.6,"color":RED})
    ax.text((boundary+len(d)-.5)/2,.9014,"Secondary comparisons >1,000 CpGs",
            ha="center",va="bottom",fontsize=7.0,color=RED)
    # All 16 evaluated sizes keep a tick mark; only LABELLED_PANELS carry a label,
    # otherwise the rotated labels collide between 900 and 2,000 CpGs.
    sizes=d.panel_size.to_numpy()
    major=[i for i,v in enumerate(sizes) if v in LABELLED_PANELS]
    ax.set_xticks(major,[f"{sizes[i]:,}" for i in major],rotation=45,ha="right")
    ax.xaxis.set_minor_locator(FixedLocator(list(x)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x",which="minor",length=2.4,width=.6,color=GREY)
    ax.set(ylim=(.90,.95),xlabel="CpG panel size",ylabel="Patient-level macro-F1 (95% CI)")
    clean(ax); ax.grid(axis="y",color=LIGHT,lw=.7); label(ax,"a")
    ax.set_title("Patient-level panel-size curve",loc="left",fontweight="bold")
    ax=axes[1]; metrics=["macro_f1","balanced_accuracy","accuracy"]
    for i,m in enumerate(metrics):
        for p,c,off in [(500,BLUE,-.13),(1000,ORANGE,.13)]:
            r=primary[(primary.panel_size==p)&(primary.metric==m)].iloc[0]; v=r.estimate_mean_across_repeats
            ax.errorbar(i+off,v,yerr=[[v-r.ci_low],[r.ci_high-v]],fmt="o",color=c,capsize=2,ms=4,label=f"{p:,} CpG" if i==0 else None)
    ax.set_xticks(range(3),["Macro-F1","Balanced\naccuracy","Accuracy"]); ax.set_ylim(.90,.97); ax.set_ylabel("Patient-level performance (95% CI)"); clean(ax); ax.legend(frameon=False,fontsize=7.3,loc="upper left"); label(ax,"b"); ax.set_title("Primary compact panels",loc="left",fontweight="bold")
    ci_lo=f"{paired.ci_low:.4f}".replace("-","−")
    ax.text(.02,.02,rf"$\Delta$ macro-F1 = {paired.estimate:.4f}"+f"\n95% CI, {ci_lo} to {paired.ci_high:.4f}",transform=ax.transAxes,fontsize=7.2,va="bottom")
    ax=axes[2]; metrics=["Nogueira","Jaccard","RBO"]
    summary={500:[.673,.953,.971],1000:[.685,.970,.971]}
    x=np.arange(3)
    a=ax.bar(x-.19,summary[500],.30,color=BLUE,label="500 CpG")
    b=ax.bar(x+.19,summary[1000],.30,color=ORANGE,label="1,000 CpG")
    # Staggered label heights: at a bar-centre spacing of 0.38 axis units the two
    # labels of a pair would otherwise touch (e.g. "0.9710.971").
    for bars,dy in ((a,.020),(b,.052)):
        for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+dy,f"{bar.get_height():.3f}",ha="center",va="bottom",fontsize=7.0)
    ax.set_xticks(x,metrics); ax.set_ylim(0,1.13); ax.set_ylabel("Stability across 15 folds"); clean(ax); ax.legend(frameon=False,fontsize=7.3,loc="lower center",bbox_to_anchor=(.5,-.35),ncol=2); label(ax,"c"); ax.set_title("Feature reproducibility",loc="left",fontweight="bold")
    # The per-fold recurrence counts (465/500 and 958/1,000) are stated in the
    # Results text and the figure caption; an in-panel annotation cannot avoid
    # either the tall RBO bars or the staggered value labels at this panel width.
    for ax in axes:
        ax.title.set_fontsize(9.2)
        ax.xaxis.label.set_fontsize(8.4)
        ax.yaxis.label.set_fontsize(8.4)
        ax.tick_params(labelsize=7.8)
    fig.subplots_adjust(left=.11,right=.97,top=.94,bottom=.13); save(fig,"figure2_internal_stability")


def fig3():
    frozen=pd.read_csv(ASSETS/"table_external_frozen_endpoints_ci.tsv",sep="\t")
    epic=pd.read_csv(ROOT/"results"/"manuscript_tables"/"epic_transfer_figure_source.tsv",sep="\t")
    assert len(epic)==12 and set(epic.panel_size)=={500,1000}
    assert (epic.n_correct/epic.n_patients-epic.recall).abs().max()<0.0001
    fig=plt.figure(figsize=(7.15,5.3))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.12],width_ratios=[1.12,1],hspace=.52,wspace=.65)
    order={"GSE56044":0,"GSE53051":1,"GSE48684":2,"GSE105260":3}
    rows=frozen.assign(_order=frozen.cohort.map(order)).sort_values(["_order","panel_size"],ascending=[True,True]).reset_index(drop=True)
    ax=fig.add_subplot(gs[0,0])
    ax.axhspan(-.5,1.5,color=BLUE_TINT,zorder=0)
    ax.axhspan(1.5,3.5,color=ORANGE_TINT,zorder=0)
    ax.axhspan(3.5,7.5,color=LIGHT,zorder=0)
    for yline in (1.5,3.5): ax.axhline(yline,color="white",lw=2,zorder=1)
    endpoints={"GSE56044":"33-class prediction","GSE53051":"Six-state prediction","GSE48684":"Positive-class recall","GSE105260":"Positive-class recall"}
    for i,r in rows.iterrows():
        c=BLUE if r.panel_size==500 else ORANGE
        ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.exact_binomial_95ci_low],[r.exact_binomial_95ci_high-r.estimate]],fmt="o",color=c,capsize=2,ms=4)
        ax.text(.605,i,f"{r.successes}/{r.n_samples}",ha="left",va="center",fontsize=5.5,color=GREY)
    labs=[f"{r.cohort} | {int(r.panel_size):,} CpG\n{endpoints[r.cohort]}" for _,r in rows.iterrows()]
    ax.set_yticks(range(len(rows)),labs); ax.tick_params(axis="y",labelsize=5.6); ax.invert_yaxis()
    ax.set(xlim=(.60,1.00),xlabel="Cohort-specific result (95% CI)"); clean(ax); ax.grid(axis="x",color=LIGHT,lw=.7); label(ax,"a"); ax.set_title("Independent 450K cohorts",loc="left",fontweight="bold")
    ax=fig.add_subplot(gs[0,1])
    # Four distinct hue/marker combinations instead of two colours, and the legend
    # is placed in the empty upper-left quadrant (the curves only rise above
    # ~0.05 risk beyond coverage 0.85) so that it no longer covers GSE53051-500.
    curves=[("GSE56044","o",500,BLUE,"-"),("GSE56044","o",1000,BLUE_LIGHT,"-"),
            ("GSE53051","s",500,ORANGE,"-"),("GSE53051","s",1000,ORANGE_LIGHT,"--")]
    for cohort,marker,p,c,ls in curves:
        rc=pd.read_csv(FROZEN_EXTERNAL(cohort),sep="\t")
        d=rc[rc.panel_size==p].sort_values("coverage")
        ax.plot(d.coverage,d.risk_retained,color=c,ls=ls,lw=1.15,marker=marker,ms=2.6,
                label=f"{cohort}, {p:,} CpG")
    ax.set(xlim=(.70,1.02),ylim=(0,.16),xlabel="Coverage",ylabel="Error risk among retained samples"); clean(ax); ax.legend(frameon=False,fontsize=5.1,ncol=1,loc="upper left",handlelength=1.7,labelspacing=.35,borderaxespad=.25); label(ax,"b"); ax.set_title("Confidence-threshold risk–coverage profiles",loc="left",fontweight="bold")
    lower=gs[1,:].subgridspec(2,1,height_ratios=[3,1],hspace=.25)
    ax=fig.add_subplot(lower[0,0])
    cohort_order=["GSE121377","GSE136380","GSE144487","GSE148766","GSE164269","GSE133556"]
    short={"GSE121377":"THCA","GSE136380":"LIHC","GSE144487":"SKCM",
           "GSE148766":"COAD/READ","GSE164269":"MESO","GSE133556":"OV"}
    x=np.arange(len(cohort_order))
    ax.axvspan(4.5,5.5,color=PALE_ORANGE,zorder=0)
    for p,c,offset in ((500,BLUE,-.13),(1000,ORANGE,.13)):
        d=epic[epic.panel_size==p].set_index("cohort").loc[cohort_order]
        y=d.recall.to_numpy()
        lo=y-d.ci_low.to_numpy(); hi=d.ci_high.to_numpy()-y
        ax.errorbar(x+offset,y,yerr=[lo,hi],fmt="o",color=c,ecolor=c,ms=4,
                    capsize=2.4,lw=1,label=f"{p:,} CpG",zorder=3)
    ax.set(xlim=(-.45,5.45),ylim=(-.04,1.10),ylabel="Lineage recall (95% CI)")
    ax.set_xticks([]); ax.grid(axis="y",color=LIGHT,lw=.7); clean(ax)
    ax.legend(frameon=False,ncol=2,fontsize=6.2,loc="lower left",bbox_to_anchor=(.01,.03))
    label(ax,"c"); ax.set_title("Cross-platform transfer varies across EPIC cohorts",loc="left",fontweight="bold")
    cov=fig.add_subplot(lower[1,0],sharex=ax)
    for p,c,offset in ((500,BLUE,-.13),(1000,ORANGE,.13)):
        d=epic[epic.panel_size==p].set_index("cohort").loc[cohort_order]
        cov.scatter(x+offset,100*d.coverage,s=19,color=c,zorder=3)
    cov.set(ylim=(79,100),ylabel="Coverage (%)")
    cov.set_xticks(x,[f"{short[cohort]} (n={int(epic[epic.cohort==cohort].n_patients.iloc[0])})\n{cohort}"
                      for cohort in cohort_order],fontsize=6)
    cov.set_yticks([80,90,100]); cov.grid(axis="y",color=LIGHT,lw=.6); clean(cov)
    ax.tick_params(axis="x",which="both",labelbottom=False)
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.10)
    save(fig,"figure3_external_validation")


def FROZEN_EXTERNAL(cohort):
    folder="GSE105260_official_series" if cohort=="GSE105260" else cohort
    return ROOT/"server_results_20260728"/"results"/"external_validation"/folder/"lr_panel_frozen_20260721"/"external_risk_coverage.tsv"


def error_data():
    raw=pd.read_csv(LOCKED/"repeated_cross_fitting_ensemble_confusion_matrices.tsv",sep="\t")
    labels=[c for c in raw.columns if c.startswith("TCGA-")]
    mat=raw[raw.panel_size==1000].set_index("true_label").loc[labels,labels].to_numpy(float)
    return labels,mat,mat/mat.sum(axis=1,keepdims=True)


def fig4():
    # Manuscript Figure 4 (class-level errors) and the supplementary full matrix.
    labels,mat,norm=error_data(); short=[x.replace("TCGA-","") for x in labels]
    f1=[]; support=mat.sum(1)
    for i in range(33):
        tp=mat[i,i]; pr=tp/mat[:,i].sum(); re=tp/support[i]; f1.append(2*pr*re/(pr+re))
    fig=plt.figure(figsize=(7.15,5.2)); gs=fig.add_gridspec(2,2,width_ratios=[1.05,1],height_ratios=[1,1])
    order=np.argsort(f1); ax=fig.add_subplot(gs[:,0]); ypos=np.arange(33)
    highlighted={"READ","ESCA","COAD","STAD","LGG","GBM","LUAD","LUSC"}
    point_colours=[ORANGE if short[i] in highlighted else BLUE for i in order]
    ax.scatter(np.array(f1)[order],ypos,s=18,c=point_colours,alpha=.95)
    # Class support is folded into the tick labels: as free-standing text the 33
    # "n=" strings sat on top of the markers and were hard to read.
    ax.set_yticks(ypos,[f"{short[i]} (n={int(support[i])})" for i in order],fontsize=5.2); ax.set(xlim=(0,1.05),xlabel="Class F1",ylabel="TCGA class"); ax.grid(axis="x",color=LIGHT,lw=.7); ax.spines[["top","right"]].set_visible(False); label(ax,"a"); ax.set_title("Class-level performance",loc="left",fontweight="bold")
    for tick,i in zip(ax.get_yticklabels(),order):
        if short[i] in highlighted: tick.set_color(ORANGE)
    pairs=[]
    for i in range(33):
        for j in range(33):
            if i != j and mat[i,j] > 0:
                pairs.append((int(mat[i,j]),norm[i,j],short[i],short[j]))
    pairs=sorted(pairs,key=lambda x:(x[0],x[1]),reverse=True)[:8]
    ax=fig.add_subplot(gs[0,1]); y=np.arange(8)[::-1]
    bars=ax.barh(y,[v[0] for v in pairs],color=ORANGE)
    ax.set_yticks(y,[f"{a} → {b}" for _,_,a,b in pairs],fontsize=6)
    for bar,(n,p,_,_) in zip(bars,pairs):
        ax.text(n+.7,bar.get_y()+bar.get_height()/2,f"{n} ({p:.1%})",va="center",fontsize=5.5,color=BLACK)
    ax.set(xlim=(0,max(v[0] for v in pairs)*1.42),xlabel="Misclassified patients (n)")
    clean(ax); label(ax,"b"); ax.set_title("Largest directional errors",loc="left",fontweight="bold")
    related=["ESCA","STAD","COAD","READ","LGG","GBM","LUAD","LUSC"]
    idx=[short.index(v) for v in related]
    ax=fig.add_subplot(gs[1,1]); im=ax.imshow(norm[np.ix_(idx,idx)],vmin=0,vmax=1,cmap=CMAP_SEQ,aspect="auto")
    ax.set_xticks(range(8),related,rotation=45,ha="right",fontsize=6)
    ax.set_yticks(range(8),related,fontsize=6); ax.set(xlabel="Predicted class",ylabel="Reference class")
    for start in (0,2,4,6):
        ax.add_patch(Rectangle((start-.5,start-.5),2,2,fill=False,
                               edgecolor=ORANGE,linewidth=1.2))
    label(ax,"c"); ax.set_title("Related-class error structure",loc="left",fontweight="bold")
    fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label="Within-class proportion")
    fig.subplots_adjust(hspace=.50,wspace=.60,left=.13,right=.96,top=.95,bottom=.11); save(fig,"figure4_class_errors")

    fig,ax=plt.subplots(figsize=(7.15,6.6)); im=ax.imshow(norm,vmin=0,vmax=1,cmap=CMAP_SEQ,aspect="auto")
    ax.set_xticks(range(33),short,rotation=90,fontsize=5.2); ax.set_yticks(range(33),short,fontsize=5.2)
    ax.set(xlabel="Predicted class",ylabel="Reference class")
    ax.set_title("Row-normalised 33-class confusion matrix",loc="left",fontweight="bold")
    fig.colorbar(im,ax=ax,fraction=.045,pad=.02,label="Within-class proportion")
    fig.subplots_adjust(left=.12,right=.95,top=.96,bottom=.12); save(fig,"supp_figure5_full_confusion")


def fig5():
    # Manuscript Figure 5: computational perturbations and normal-tissue calls.
    fig,axes=plt.subplots(2,2,figsize=(7.15,4.8));
    ax=axes[0,0]
    for p,c in [(500,BLUE),(1000,ORANGE)]:
        d=pd.read_csv(ROBUST/f"consensus_locked_{p}_cpg"/"robustness_seed_summary.tsv",sep="\t"); d=d[(d.condition=="added_normal_profile_fraction")&(d.metric=="macro_f1")].sort_values("level")
        ax.fill_between(d.level,d.seed_q_0_025,d.seed_q_0_975,color=c,alpha=.13); ax.plot(d.level,d.seed_median,"o-",color=c,ms=3,label=f"{p:,} CpG")
    ax.axvline(.4,color=GREY,lw=.8,ls="--"); ax.set(xlabel="Added same-project normal-profile fraction",ylabel="Seed-median macro-F1",ylim=(.58,1.02)); ax.legend(frameon=False,fontsize=6,loc="center left"); clean(ax); label(ax,"a"); ax.set_title("Fitted-model normal-admixture sensitivity",loc="left",fontweight="bold")
    # The unperturbed point is an in-sample reference, not an out-of-sample result.
    ax.text(.975,.965,"$\\lambda$ = 0: in-sample reference\nmodels fitted on all 9,065 tumours",transform=ax.transAxes,ha="right",va="top",fontsize=5.0,color=GREY,linespacing=1.35,bbox={"facecolor":"white","edgecolor":LIGHT,"linewidth":.5,"pad":1.8,"alpha":.95})
    ax.text(.03,.06,"$\\lambda$ = 0.4: $\\Delta$ macro-F1\n500: −0.0722   1,000: −0.0686",
            transform=ax.transAxes,ha="left",va="bottom",fontsize=5.4,color=BLACK,
            bbox={"facecolor":"white","edgecolor":LIGHT,"linewidth":.5,"pad":1.6,"alpha":.95})
    ax=axes[0,1]
    for p,c,pos in [(500,BLUE,1),(1000,ORANGE,2)]:
        panel_dir=OOF_PROBE_LOSS/f"consensus_locked_{p}_cpg"
        d=pd.read_csv(panel_dir/"oof_probe_loss_confirmatory_deltas.tsv",sep="\t"); v=d.delta_macro_f1
        ci=pd.read_csv(panel_dir/"oof_probe_loss_confirmatory_patient_bootstrap_ci.tsv",sep="\t").iloc[0]
        ax.scatter(np.random.default_rng(p).normal(pos,.035,len(v)),v,s=8,color=c,alpha=.48)
        ax.boxplot(v,positions=[pos],widths=.45,showfliers=False,patch_artist=True,
                   boxprops={"facecolor":"white","edgecolor":c},medianprops={"color":BLACK})
        ax.errorbar(pos,ci.estimate_seed_median_across_seeds,
                    yerr=[[ci.estimate_seed_median_across_seeds-ci.ci_low],
                          [ci.ci_high-ci.estimate_seed_median_across_seeds]],
                    fmt="D",color=c,ms=3.2,capsize=2.5,zorder=5)
    ax.axhline(0,color=GREY,lw=.8,ls="--")
    ax.set_xticks([1,2],["500 CpG","1,000 CpG"]); ax.set(ylabel=r"$\Delta$ macro-F1 after 20% probe loss",ylim=(-.033,.003)); clean(ax); label(ax,"b"); ax.set_title("Out-of-fold probe-loss robustness",loc="left",fontweight="bold")
    ax.text(.98,.05,"−0.0232     −0.0143",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.4,color=BLACK)
    norm=pd.read_csv(ASSETS/"table_normal_ood_summary.tsv",sep="\t").sort_values("panel_size"); ax=axes[1,0]; x=np.arange(2)
    bars_500=ax.bar(x-.18,[norm.high_confidence_cancer_call_rate.iloc[0],norm.organ_lineage_concordance.iloc[0]],.36,color=BLUE,label="500 CpG")
    bars_1000=ax.bar(x+.18,[norm.high_confidence_cancer_call_rate.iloc[1],norm.organ_lineage_concordance.iloc[1]],.36,color=ORANGE,label="1,000 CpG")
    for bars in (bars_500,bars_1000):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.016,
                    f"{100*bar.get_height():.1f}%",ha="center",va="bottom",fontsize=5.2)
    # Both bars use all 747 retained normal samples; only the left bar additionally
    # applies the 0.90 confidence threshold, so the panel title no longer implies a
    # threshold denominator for the organ-lineage bar.
    ax.set_xticks(x,["High-confidence\ntumour-class call\n(confidence $\\geq$ 0.90)","Organ-lineage\nconcordance\n(all samples)"]); ax.set(ylim=(0,1.02),ylabel="Proportion of normal tissues"); clean(ax); ax.legend(frameon=False,fontsize=5.6,loc="upper left"); label(ax,"c"); ax.set_title("Normal-tissue stress test (n = 747)",loc="left",fontweight="bold")
    n_norm=int(norm.n_normal_samples.iloc[0]) if "n_normal_samples" in norm.columns else 747
    ax.text(.5,-.34,f"Both bars use all {n_norm} retained normal tissues.",transform=ax.transAxes,ha="center",va="top",fontsize=5.0,color=GREY)
    ax=axes[1,1]
    vals=[]
    for p in (500,1000):
        d=pd.read_csv(ROBUST/f"consensus_locked_{p}_cpg"/"normal_ood_predictions.tsv.gz",sep="\t"); vals.append(d.max_probability.to_numpy())
    bp=ax.boxplot(vals,positions=[0,1],widths=.5,showfliers=False,patch_artist=True,boxprops={"facecolor":"white","edgecolor":BLACK},medianprops={"color":BLACK,"linewidth":1.1},whiskerprops={"color":BLACK},capprops={"color":BLACK})
    for x,v,c in zip([0,1],vals,[BLUE,ORANGE]): ax.scatter(np.random.default_rng(int(x)+9).normal(x,.035,len(v)),v,s=2.2,color=c,alpha=.16)
    ax.axhline(.90,color=GREY,lw=.8,ls="--",zorder=0)
    ax.set_xticks([0,1],["500 CpG","1,000 CpG"]); ax.set(ylim=(0,1.02),ylabel="Maximum confidence score"); clean(ax); label(ax,"d"); ax.set_title("Distribution of normal-tissue confidence",loc="left",fontweight="bold")
    fig.subplots_adjust(hspace=.58,wspace=.40); save(fig,"figure5_robustness_normal")


def supp_annotation():
    stable=pd.read_csv(CURRENT/"stable_cpg_panel_annotated.tsv",sep="\t"); cls=pd.read_csv(CURRENT/"class_specific_top_cpg_annotated.tsv",sep="\t")
    top=stable[stable.panel_size==1000].sort_values(["selected_count","mean_rank"],ascending=[False,True]).head(100)
    genes=cls[cls.gene_symbol.fillna("").ne("")].groupby("gene_symbol").mean_abs_coef.mean().nlargest(12).sort_values(ascending=False)
    classes=["TCGA-BRCA","TCGA-COAD","TCGA-READ","TCGA-KIRC","TCGA-KIRP","TCGA-LUAD","TCGA-LUSC","TCGA-PAAD","TCGA-THCA"]
    heat=cls[cls.gene_symbol.isin(genes.index)&cls.label.isin(classes)].pivot_table(index="gene_symbol",columns="label",values="mean_coef",aggfunc="mean",fill_value=0).reindex(index=genes.index,columns=classes).fillna(0)
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.25),gridspec_kw={"width_ratios":[.85,1.35]}); ax=axes[0]; context=pd.Series({"Promoter":top.is_promoter.fillna(False).mean(),"Island":top.relation_to_cpg_island.fillna("").eq("Island").mean(),"Shore":top.relation_to_cpg_island.fillna("").str.contains("Shore").mean(),"Gene annotated":top.gene_symbol.fillna("").ne("").mean()}); ax.barh(context.index,context.values,color=BLUE); ax.set(xlim=(0,1),xlabel="Proportion of recurrent top-100 CpGs"); clean(ax); label(ax,"a"); ax.set_title("Annotation context",loc="left",fontweight="bold")
    ax=axes[1]; im=ax.imshow(heat,cmap=CMAP_DIV,vmin=-1,vmax=1,aspect="auto"); ax.set_xticks(range(len(classes)),[x.replace("TCGA-","") for x in classes],rotation=35,ha="right"); ax.set_yticks(range(len(genes)),genes.index); fig.colorbar(im,ax=ax,fraction=.045,pad=.02,label="Mean CpG coefficient"); label(ax,"b"); ax.set_title("CpG-associated gene annotations",loc="left",fontweight="bold")
    fig.subplots_adjust(wspace=.55,bottom=.20); save(fig,"supp_figure3_annotation")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); supp_annotation()
    # Rebuild the scientific workflow last so the older card-based draft above
    # cannot overwrite the submission version.
    from build_figure1_scientific_workflow import OUT as NPJ_FIG1_OUT, main as build_npj_figure1
    build_npj_figure1()
    for suffix in ("pdf", "png"):
        shutil.copy2(
            NPJ_FIG1_OUT / f"figure1_study_design.{suffix}",
            OUT / f"figure1_study_design.{suffix}",
        )
    print(f"Wrote final submission figures to {OUT}")

if __name__ == "__main__": main()
