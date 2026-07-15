# -*- coding: utf-8 -*-
"""
main.py
=======
Demonstration script for the KNN vs NC Research Framework.

A Critique of:
    "Comparative Analysis Between the Traditional and Nearest Centroid
     Methods in the K-Nearest Neighbor Algorithm"
    Mladenova & Valova, IEEE HORA 2022

This file shows the clean API by composing objects and calling methods.
There is no CLI, no argparse, no menus — just a readable script.

Run:
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# ── Datasets ──────────────────────────────────────────────────────────────────
from datasets import IrisDataset, WineDataset, IonosphereDataset

# ── Statistical tests ─────────────────────────────────────────────────────────
from statistical_tests import (
    NormalityTest,
    UnimodalityTest,
    SkewnessTest,
    KurtosisTest,
    SeparabilityTest,
)

# ── Visualisations ────────────────────────────────────────────────────────────
from visualization import (
    Histogram,
    QQPlot,
    PCAScatter,
    BoxPlot,
    DensityPlot,
    SummaryHeatmap,
)

# ── Synthetic generators ──────────────────────────────────────────────────────
from synthesis import (
    CentralTendencyGenerator,
    NormalGenerator,
    MultimodalGenerator,
    SkewedGenerator,
    UniformGenerator,
    HeavyTailGenerator,
    ImbalancedGenerator,
    NonCentralGenerator,
    MixedGenerator,
)

# ── Experiments ───────────────────────────────────────────────────────────────
from experiments import KNNvsNCExperiment, DistributionSweepExperiment





# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Load Real Datasets
# ══════════════════════════════════════════════════════════════════════════════

iris     = IrisDataset()
df_iris  = iris.load()
print("\nIris:", iris.summary())

wine     = WineDataset()
df_wine  = wine.load()
print("Wine:", wine.summary())

iono     = IonosphereDataset()
df_iono  = iono.load()
print("Ionosphere:", iono.summary())


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Statistical Tests (Iris, per-class)
# ══════════════════════════════════════════════════════════════════════════════


for cls in iris.class_labels():
    subset = iris.class_subset(cls)
    print(f"\n--- Class: {cls} (n={len(subset)}) ---")

    # Normality
    r = NormalityTest(subset, alpha=0.05).run_test()
    print(f"  Normality    HZ={r['hz_statistic']}  p={r['p_value']}  "
          f"Normal={r['is_normal']}")

    # Unimodality
    r = UnimodalityTest(subset, alpha=0.05).run_test()
    print(f"  Unimodality  {r['pct_unimodal']}% of features unimodal")

    # Skewness
    r = SkewnessTest(subset, threshold=1.0).run_test()
    print(f"  Skewness     {r['pct_near_normal']}% near-normal  "
          f"→ {r['overall_interpretation']}")

    # Kurtosis
    r = KurtosisTest(subset, threshold=2.0).run_test()
    print(f"  Kurtosis     {r['pct_near_normal']}% near-normal  "
          f"→ {r['overall_interpretation']}")

# Separability — needs full df with labels
r = SeparabilityTest(df_iris, target_col="target").run_test()
print(f"\nSeparability (all class pairs):")
print(f"  Avg ratio: {r['avg_ratio']}")
print(f"  → {r['overall_interpretation']}")
for pair in r["pair_results"]:
    print(f"    {pair['interpretation']}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Visualisations (Iris)
# ══════════════════════════════════════════════════════════════════════════════

print("\nHistogram + KDE ...")
Histogram(df_iris, target_col="target",
          title="Iris — Histogram + KDE", bins=12).draw()

print("Density Plot ...")
DensityPlot(df_iris, target_col="target",
            title="Iris — KDE Density").draw()

print("Q-Q Plots ...")
QQPlot(df_iris, target_col="target",
       title="Iris — Q-Q Plots").draw()

print("PCA 2D Scatter ...")
PCAScatter(df_iris, target_col="target",
           title="Iris — PCA 2D").draw()

print("Box Plots ...")
BoxPlot(df_iris, target_col="target",
        title="Iris — Box Plots").draw()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Synthetic Data Generation
# ══════════════════════════════════════════════════════════════════════════════


generators = [
    ("Central Tendency",   CentralTendencyGenerator(n_samples=600, spread=0.3,   separation=8.0)),
    ("Normal",             NormalGenerator(n_samples=600,          separation=4.0)),
    ("Uniform",            UniformGenerator(n_samples=600,         class_width=4.0)),
    ("Skewed",             SkewedGenerator(n_samples=600,          skewness=1.5)),
    ("Heavy-Tailed",       HeavyTailGenerator(n_samples=600,       degrees_of_freedom=2)),
    ("Multimodal",         MultimodalGenerator(n_samples=600,      modes_per_class=2)),
    ("Imbalanced",         ImbalancedGenerator(n_samples=600,      imbalance_ratio=5.0)),
    ("Non-Central",        NonCentralGenerator(n_samples=600,      spread=3.5, separation=1.5)),
    ("Mixed Features",     MixedGenerator(n_samples=600,           n_features=6)),
]

synthetic_dfs = {}
for name, gen in generators:
    df = gen.generate()
    synthetic_dfs[name] = df
    counts = df["target"].value_counts().to_dict()
    print(f"  {name:<22} shape={df.shape}  classes={counts}")

# Show PCA scatter for Central Tendency vs Non-Central side by side
print("\nPCA — Central Tendency (NC ideal):")
PCAScatter(synthetic_dfs["Central Tendency"],
           title="Central Tendency — PCA 2D").draw()

print("PCA — Non-Central (NC should fail):")
PCAScatter(synthetic_dfs["Non-Central"],
           title="Non-Central (Scattered) — PCA 2D").draw()

print("PCA — Multimodal:")
PCAScatter(synthetic_dfs["Multimodal"],
           title="Multimodal — PCA 2D").draw()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — KNN vs NC on Individual Scenarios
# ══════════════════════════════════════════════════════════════════════════════


key_scenarios = [
    ("Central Tendency (NC ideal)",    synthetic_dfs["Central Tendency"]),
    ("Normal",                         synthetic_dfs["Normal"]),
    ("Multimodal (NC broken)",         synthetic_dfs["Multimodal"]),
    ("Skewed",                         synthetic_dfs["Skewed"]),
    ("Non-Central (NC worst case)",    synthetic_dfs["Non-Central"]),
    ("Ionosphere (real, from paper)",  df_iono),
]

summary_rows = []

for label, df in key_scenarios:
    exp    = KNNvsNCExperiment(df, k=5, cv_folds=10)
    result = exp.run()

    knn_acc = result["knn_summary"]["accuracy_mean"]
    nc_acc  = result["nc_summary"]["accuracy_mean"]
    knn_f1  = result["knn_summary"]["f1_mean"]
    nc_f1   = result["nc_summary"]["f1_mean"]
    gap     = round(knn_acc - nc_acc, 4)
    winner  = "KNN" if gap > 0.001 else ("NC" if gap < -0.001 else "Tie")

    print(f"\n  {label}")
    print(result["comparison_df"].to_string())

    summary_rows.append({
        "Scenario":         label,
        "KNN Acc":          knn_acc,
        "NC Acc":           nc_acc,
        "KNN F1":           knn_f1,
        "NC F1":            nc_f1,
        "Acc Gap (KNN-NC)": gap,
        "Winner":           winner,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Full Distribution Sweep
# ══════════════════════════════════════════════════════════════════════════════

sweep   = DistributionSweepExperiment(n_samples=600, n_features=4,
                                       n_classes=2, k=5, cv_folds=10)
results = sweep.run()

print("\n--- Full Sweep Comparison Table ---")
print(results["comparison"].to_string())

wc = results["winner_counts"]
print(f"\nWinner counts across all scenarios:")
print(f"  KNN wins : {wc['KNN']}")
print(f"  NC wins  : {wc['NC']}")
print(f"  Ties     : {wc['Tie']}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Summary Visualisation
# ══════════════════════════════════════════════════════════════════════════════

sweep_df = results["comparison"]

SummaryHeatmap(
    sweep_df,
    heatmap_cols=["KNN Acc", "NC Acc", "KNN F1", "NC F1"],
    bar_col="Acc Gap (KNN-NC)",
    title="KNN vs NC — Full Distribution Sweep Summary",
    vmin=0.0,
    vmax=1.0,
).draw()

print("\nAll sections complete.")
