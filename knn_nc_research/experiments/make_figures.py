"""Builds every thesis figure from the CSV files in results/."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

from src.paths import RESULTS_DIR, FIGURES_DIR

TEAL, ORANGE, GREY, BLUE = "#005f5f", "#d9822b", "#8a8a8a", "#3a6ea5"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)


def fig_pipeline():
    """Figure 2.1: the five stages of the study."""
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.1, "Stage 1\nReplication", "15 UCI datasets,\nraw features,\nKNN vs NCN\n(k = 5)"),
        (2.1, "Stage 2\nScaling test", "same folds,\nStandardScaler\nfit on train fold"),
        (4.1, "Stage 3\nMechanism", "heterogeneity\nindex, synthetic\nscale sweep,\nk sweep"),
        (6.1, "Stage 4\nStructure", "skewness and\nseparability,\nNC as a global\nreference"),
        (8.1, "Stage 5\nGuidance", "evidence-graded\nselection rules"),
    ]
    for x, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.25), 1.75, 2.5, boxstyle="round,pad=0.03",
                                    fc="#e6f0f0", ec=TEAL, lw=1.1))
        ax.text(x + 0.875, 2.3, title, ha="center", va="center", fontsize=8.5,
                weight="bold", color=TEAL)
        ax.text(x + 0.875, 1.15, body, ha="center", va="center", fontsize=7.2, linespacing=1.3)
    for x in (1.9, 3.9, 5.9, 7.9):
        ax.annotate("", xy=(x + 0.2, 1.5), xytext=(x - 0.02, 1.5),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.3))
    save(fig, "fig_pipeline.pdf")


def fig_gap_bars(real):
    """Figure 5.1: NCN - KNN gap, raw vs standardised."""
    d = real.sort_values("gap_raw", ascending=False)
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(x - w / 2, d.gap_raw, w, color=ORANGE, label="Raw features (original protocol)")
    ax.bar(x + w / 2, d.gap_std, w, color=TEAL, label="Standardised features")
    for i, (_, r) in enumerate(d.iterrows()):
        if r.sig_raw:
            ax.text(i - w / 2, r.gap_raw + 0.004, "*", ha="center", color=ORANGE, fontsize=11)
        if r.sig_std:
            ax.text(i + w / 2, r.gap_std + 0.004, "*", ha="center", color=TEAL, fontsize=11)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(d.dataset, rotation=40, ha="right")
    ax.set_ylabel("Accuracy gap (NCN $-$ KNN)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig_gap_scaled_unscaled.pdf")


def fig_heterogeneity(real):
    """Figure 5.2: gap shrinkage against scale heterogeneity."""
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.scatter(real.heterogeneity, real.shrinkage, s=28, color=TEAL, zorder=3)
    offsets = {"Lung cancer": (6, -3), "Ionosphere": (-10, 7), "Lenses": (5, 4), "Glass": (5, -8)}
    labelled = ("Wine", "Stars", "Cryotherapy", "Ionosphere", "Breast cancer",
                "Pima diabetes", "Lenses", "Lung cancer", "Glass")
    for _, r in real[real.dataset.isin(labelled)].iterrows():
        ax.annotate(r.dataset, (r.heterogeneity, r.shrinkage),
                    xytext=offsets.get(r.dataset, (4, 3)), textcoords="offset points", fontsize=7)
    b, a = np.polyfit(real.heterogeneity, real.shrinkage, 1)
    xs = np.linspace(0, real.heterogeneity.max() + 0.2, 50)
    ax.plot(xs, a + b * xs, "--", color=GREY, lw=1)
    rho, p = stats.spearmanr(real.heterogeneity, real.shrinkage)
    ax.text(0.03, 0.95, rf"Spearman $\rho$ = {rho:+.2f}, p = {p:.3f}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Scale heterogeneity  $\log_{10}(\sigma_{max}/\sigma_{min})$")
    ax.set_ylabel("Gap shrinkage (raw gap $-$ scaled gap)")
    save(fig, "fig_heterogeneity_shrinkage.pdf")


def fig_mechanism(sweep, ksweep):
    """Figure 5.3: synthetic scale sweep and Wine k sweep."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a1.plot(sweep.scale, sweep.gap_raw, "o-", color=ORANGE, label="Raw features")
    a1.plot(sweep.scale, sweep.gap_std, "s--", color=TEAL, label="Standardised")
    a1.set_xscale("log"); a1.axhline(0, color="k", lw=0.5)
    a1.set_xlabel("Scale factor of the weak feature"); a1.set_ylabel("NCN $-$ KNN")
    a1.set_title("(a) Synthetic scale sweep", fontsize=9); a1.legend(frameon=False)
    a2.plot(ksweep.k, ksweep.gap_raw, "o-", color=ORANGE, label="Raw features")
    a2.plot(ksweep.k, ksweep.gap_std, "s--", color=TEAL, label="Standardised")
    a2.axhline(0, color="k", lw=0.5)
    a2.set_xlabel("Neighbourhood size $k$"); a2.set_ylabel("NCN $-$ KNN")
    a2.set_title("(b) Wine, $k$ sweep", fontsize=9); a2.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig_mechanism.pdf")


def fig_threeway(prof):
    """Figure 5.4: three-way accuracy and gap against skewness."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(len(prof)); w = 0.27
    a1.bar(x - w, prof.knn, w, color=BLUE, label="KNN")
    a1.bar(x, prof.ncn, w, color=ORANGE, label="NCN")
    a1.bar(x + w, prof.nc, w, color=TEAL, label="NC")
    a1.set_xticks(x); a1.set_xticklabels(prof.dataset, rotation=45, ha="right", fontsize=7)
    a1.set_ylabel("Accuracy (standardised)"); a1.set_ylim(0, 1.05)
    a1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    a1.set_title("(a) Three-way comparison", fontsize=9, pad=18)
    a2.scatter(prof.skewness, prof.gap_nc, color=TEAL, s=22, label="NC $-$ KNN")
    a2.scatter(prof.skewness, prof.gap_ncn, color=ORANGE, marker="^", s=22, label="NCN $-$ KNN")
    t = prof[prof.dataset == "Thyroid"].iloc[0]
    a2.annotate("Thyroid", (t.skewness, t.gap_nc), xytext=(-38, 6),
                textcoords="offset points", fontsize=7)
    a2.axhline(0, color="k", lw=0.5); a2.set_xlabel("Mean |skewness|")
    a2.set_ylabel("Gap vs KNN"); a2.legend(frameon=False, fontsize=7, loc=(0.38, 0.3))
    a2.set_title("(b) Gap against skewness", fontsize=9)
    fig.tight_layout(); save(fig, "fig_threeway.pdf")


def fig_skewness(s):
    """Figure 5.5: skewness with separation held fixed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a1.plot(s.skewness, s.ncn - s.knn, "^-", color=ORANGE, label="NCN $-$ KNN")
    a1.plot(s.skewness, s.nc - s.knn, "s-", color=TEAL, label="NC $-$ KNN")
    a1.axhline(0, color="k", lw=0.5)
    a1.set_xlabel("Measured mean |skewness|"); a1.set_ylabel("Accuracy gap vs KNN")
    a1.set_title("(a) Gap to KNN, separation held fixed", fontsize=9); a1.legend(frameon=False)
    a2.plot(s.skewness, s.separability, "o-", color=GREY, label="Separability")
    a2.plot(s.skewness, s.compactness, "s-", color="k", label="Compactness")
    a2.set_ylim(0, 3); a2.set_xlabel("Measured mean |skewness|"); a2.set_ylabel("Measured value")
    a2.set_title("(b) Control check", fontsize=9); a2.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig_skewness_fixed.pdf")


def run():
    read = lambda name: pd.read_csv(RESULTS_DIR / name)
    fig_pipeline()
    fig_gap_bars(read("exp1_scaling.csv"))
    fig_heterogeneity(read("exp1_scaling.csv"))
    fig_mechanism(read("exp2_scale_sweep.csv"), read("exp2_k_sweep_wine.csv"))
    fig_threeway(read("exp3_structural_profile.csv"))
    fig_skewness(read("exp4_skewness_fixed_separation.csv"))
    print(f"Figures saved in {FIGURES_DIR}")


if __name__ == "__main__":
    run()
