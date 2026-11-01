"""Render the method figure directly from the revised mathematical protocol."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42, "font.size":11})
    fig, ax = plt.subplots(figsize=(15.5,6.7))
    ax.set(xlim=(0,16),ylim=(0,7.4))
    ax.axis("off")

    def box(x,y,w,h,text,color="#f2f5f8",size=11,edge="#7a8894"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.12",
                                   facecolor=color,edgecolor=edge,linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,linespacing=1.5)

    def arrow(a,b,color="#506779"):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=12,
                                    linewidth=1.3,color=color))

    for x,w,color,title in ((.12,4.42,"#edf4fa","(a) Client-local adaptation"),
                             (4.82,5.16,"#f4f0f8","(b) Authority-scoped routing"),
                             (10.26,5.6,"#eef5f1","(c) Target-cohort aggregation")):
        box(x,.95,w,6.25,"",color,edge=color)
        ax.text(x+.15,6.92,title,fontsize=14,weight="bold",va="center")

    box(.42,5.5,3.82,.9,"Private case records and retrieval\nremain at institution $k$",size=12)
    arrow((2.33,5.43),(2.33,5.12))
    box(.42,3.95,1.72,1.1,"Shared branch\n$B_s A_s$\nshared training", "#dceaf5",size=11)
    box(2.5,3.95,1.72,1.1,"Private branch\n$B_k A_k$\nlocal training", "#fff0da",size=11)
    ax.text(2.33,4.5,"+",ha="center",fontsize=18)
    box(.42,2.32,3.82,1.13,"Disjoint client holdout\nselects private branch ($\\gamma_k$)\nExcluded from both training phases",size=10.7)
    arrow((3.36,3.88),(3.36,3.52))
    box(.42,1.27,3.82,.58,"Upload shared factors and probe summaries", "#e2edf5",size=10.5)
    arrow((4.27,1.57),(4.68,1.57))

    box(5.1,5.48,4.6,.92,"Target authority $j$, task and date\nAudited metadata + calibrated comparison",size=11.5)
    arrow((7.4,5.4),(7.4,5.1))
    box(5.1,4.25,4.6,.78,"Comparability gate\nValid issue, label space and authority mapping",size=11)
    arrow((7.4,4.16),(7.4,3.87))
    box(5.1,2.98,2.13,.8,"AuthorityLocal\nretain locally", "#fff0da",size=11)
    box(7.58,2.98,2.13,.8,"Transferable\nshared cohort", "#dbefdf",size=11)
    box(5.1,1.68,2.13,.8,"Unsupported\ndownweight / reject", "#f7e2df",size=10.5)
    box(7.58,1.68,2.13,.8,"Unresolved\nretain uncertainty", "#e5e8ef",size=10.5)
    ax.plot([9.78,10.12,10.12],[3.4,3.4,5.9],color="#506779",linewidth=1.3)
    arrow((10.12,5.9),(10.5,5.9))

    box(10.58,5.42,4.96,1.05,"Comparable-probe exposure $\\delta_{k\\to j}$\nCitation / reasoning / verdict / rule distances\nAvailable components use calibrated weights",size=10.7)
    arrow((13.06,5.35),(13.06,5.03))
    box(10.58,3.78,4.96,1.18,
        "$b_{k\\to j}\\propto u_{k\\to j}\\min(n_k,n_{\\max})$\n"
        "$\\widehat{\\alpha}_{k\\to j}\\propto b_{k\\to j}\\exp(-\\lambda\\delta_{k\\to j})$",
        "#e0eee6",size=13)
    arrow((13.06,3.7),(13.06,3.39))
    box(10.58,2.28,4.96,1.03,
        "Diversity floor, then temporal smoothing\n"
        "$\\widetilde{\\alpha}=(1-\\tau)\\widehat{\\alpha}+\\tau b$\n"
        "$\\alpha_t=(1-\\nu)\\alpha_{t-1}+\\nu\\widetilde{\\alpha}_t$",size=11.7)
    arrow((13.06,2.2),(13.06,1.92))
    box(10.58,1.22,4.96,.64,
        "$\\vartheta_{s,t+1,j}=\\sum_{k\\in\\mathcal{G}_j}\\alpha_{k\\to j}\\vartheta_{s,t,k}$", "#dbece1",size=13)

    box(.42,.12,15.12,.55,
        "Client inference: $W_k=W_0+s(B_s A_s+\\gamma_k B_k A_k)$"
        "     |     shared factors broadcast; private factors stay local", "#f7f8fa",size=13)
    fig.tight_layout(pad=.4)
    target = ROOT / "papers/ipm/Federated Legal elsevier/figures/fig2_operational_conflict_aggregation"
    fig.savefig(target.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"),dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
