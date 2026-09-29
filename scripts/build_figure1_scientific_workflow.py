#!/usr/bin/env python3
"""Draw the manuscript workflow from the frozen cohort and evaluation design."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

from _figure_style import BLACK, GREY, BLUE, ORANGE, GREEN, LIGHT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "manuscript_figures_npj_fig1"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 600, "savefig.dpi": 600})


def stage(ax, x, width, title, colour, highlight=False):
    ax.add_patch(FancyBboxPatch((x, .10), width, .66,
                                boxstyle="round,pad=.012,rounding_size=.018",
                                facecolor="white", edgecolor=colour if highlight else LIGHT,
                                linewidth=1.0 if highlight else .6))
    ax.add_patch(Rectangle((x, .69), width, .07, color=colour, linewidth=0))
    ax.text(x + .025, .655, title, fontsize=7.7, weight="bold", color=BLACK,
            ha="left", va="top")


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch((start, .48), (end, .48), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.5, color=GREY))


def people(ax, x, y, colour, cols=5, rows=2):
    for j in range(rows):
        for i in range(cols):
            cx=x+i*.026; cy=y-j*.055
            ax.add_patch(Circle((cx,cy),.009,facecolor=colour,edgecolor="none"))
            ax.plot([cx,cx],[cy-.010,cy-.033],color=colour,lw=1.7)


def methylation(ax, x, y, width, height, seed=12):
    rng=np.random.default_rng(seed)
    values=rng.choice([0,1,2],size=(6,9),p=[.32,.36,.32])
    ax.imshow(values,cmap=mpl.colors.ListedColormap(["#DCE8F3",BLUE,ORANGE]),
              interpolation="nearest",aspect="auto",
              extent=(x,x+width,y,y+height),vmin=0,vmax=2,zorder=2)
    for gx in np.linspace(x,x+width,10): ax.plot([gx,gx],[y,y+height],color="white",lw=.35,zorder=3)
    for gy in np.linspace(y,y+height,7): ax.plot([x,x+width],[gy,gy],color="white",lw=.35,zorder=3)


def heading(ax, letter, title, note):
    ax.text(.006,.985,letter,fontsize=11.5,weight="bold",va="top",color=BLACK)
    ax.text(.045,.985,title,fontsize=9.0,weight="bold",va="top",color=BLACK)
    ax.text(.045,.875,note,fontsize=6.5,va="top",color=GREY)


def main():
    fig,axes=plt.subplots(3,1,figsize=(7.15,4.55))
    for ax in axes:
        ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")

    ax=axes[0]
    heading(ax,"A","Patient-level development","Training-only preprocessing and feature ranking in every outer fold")
    stage(ax,.04,.25,"TCGA 450K tumours",BLUE)
    people(ax,.075,.57,BLUE)
    ax.text(.075,.22,"9,065 tumours · 8,923 patients\n33 lineages",fontsize=6.8,va="bottom")
    arrow(ax,.30,.355)
    stage(ax,.36,.27,"Grouped 5-fold × 3",BLUE)
    for r in range(3):
        for c in range(5):
            ax.add_patch(Rectangle((.395+c*.037,.52-r*.065),.031,.045,
                                   facecolor=ORANGE if c==r else "#DCE8F3",
                                   edgecolor="white",lw=.5))
    ax.text(.395,.22,"Patients stay together\n8,923 OOF calls/repeat",fontsize=6.8,va="bottom")
    arrow(ax,.64,.695)
    stage(ax,.70,.26,"CpG ranking",BLUE)
    methylation(ax,.73,.39,.18,.20)
    ax.text(.73,.22,"Fold-specific 500 / 1,000 CpGs",fontsize=6.8,va="bottom")

    ax=axes[1]
    heading(ax,"B","Consensus panel and locked model","Recurrent fold selections define the external model inputs")
    stage(ax,.04,.25,"15 fold selections",ORANGE)
    for i,h in enumerate([.07,.14,.20,.15,.09]):
        ax.add_patch(Rectangle((.075+i*.033,.36),.021,h,facecolor=ORANGE,edgecolor="none"))
    ax.text(.075,.22,"Recurrence + within-fold rank",fontsize=6.8,va="bottom")
    arrow(ax,.30,.355)
    stage(ax,.36,.27,"Two locked panels",ORANGE,highlight=True)
    for y,n,label in ((.52,10,"500"),(.40,16,"1,000")):
        for i in range(n):
            ax.add_patch(Rectangle((.39+i*.012,y),.009,.055,
                                   facecolor=BLUE if i%3 else ORANGE,edgecolor="none"))
        ax.text(.59,y+.027,label,fontsize=6.2,va="center")
    ax.text(.39,.22,"Compact · reference",fontsize=6.8,va="bottom")
    arrow(ax,.64,.695)
    stage(ax,.70,.26,"Final model fitting",ORANGE)
    ax.text(.735,.46,"33 lineage\nprobabilities",fontsize=7.2,weight="bold",ha="left",va="center")
    ax.text(.735,.22,"Fitted once; held fixed\nfor external evaluation",fontsize=6.8,va="bottom")

    ax=axes[2]
    heading(ax,"C","Independent evaluation and failure modes","Each external cohort is interpreted within its supported endpoint")
    for x,w,title,colour in ((.04,.28,"450K cohorts",GREEN),(.36,.28,"Processed EPIC",GREEN),
                             (.68,.28,"Stress tests",GREEN)):
        stage(ax,x,w,title,colour,highlight=title!="Stress tests")
    for i,h in enumerate([.12,.16,.10,.14]):
        ax.add_patch(Rectangle((.08+i*.038,.40),.027,h,facecolor=GREEN,edgecolor="none"))
    ax.text(.08,.22,"4 cohorts · 307 samples\n33-class / six-state / recall",fontsize=6.8,va="bottom")
    for i,v in enumerate([1.0,.96,.76,.59,.03]):
        ax.scatter(.395+i*.041,.37+v*.17,s=20,color=ORANGE if i==4 else GREEN,zorder=4)
    ax.text(.395,.22,"6 cohorts · 460 patients\npositive-lineage recall",fontsize=6.8,va="bottom")
    ax.plot([.72,.78,.84,.90],[.55,.52,.44,.37],"o-",color=ORANGE,lw=1.2,ms=3)
    ax.text(.72,.22,"Admixture · probe loss\n747 normal tissues",fontsize=6.8,va="bottom")

    fig.subplots_adjust(left=.015,right=.99,top=.99,bottom=.02,hspace=.075)
    for suffix in ("pdf","png"):
        fig.savefig(OUT/f"figure1_study_design.{suffix}",bbox_inches="tight",dpi=600)
    plt.close(fig)
    print(f"Wrote {OUT/'figure1_study_design.pdf'}")


if __name__=="__main__":
    main()
