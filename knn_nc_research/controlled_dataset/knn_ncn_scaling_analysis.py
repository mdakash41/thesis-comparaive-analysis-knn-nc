# -*- coding: utf-8 -*-
"""
knn_ncn_scaling_analysis.py
===========================
Tests whether the NCN advantage reported by Mladenova & Valova (2022)
depends on feature scaling (the original used NO preprocessing).

For every dataset:
  1. KNN vs NCN WITHOUT scaling  (replicates the original protocol)
  2. KNN vs NCN WITH StandardScaler (fit on train fold only)
  3. Corrected resampled t-test (Nadeau & Bengio, 2003) per condition
  4. Scale-heterogeneity index of the raw features

Then:
  5. Correlation: scale heterogeneity vs gap shrinkage (unscaled - scaled)
  6. Wilcoxon test across datasets: is the unscaled gap larger?
  7. Synthetic sweep: one feature's scale is increased (controlled evidence)
  8. k sweep on a chosen dataset

Outputs: console tables, results/*.csv, results/*.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Data preparation (same as your working file) ─────────────────────────────

def prepare(df, target_col, dataset_name):
    df = df.dropna().reset_index(drop=True)
    features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    X = features.values.astype(float)
    y = LabelEncoder().fit_transform(df[target_col].values)
    print(f"  {dataset_name}: n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}")
    return X, y


# ── NCN (Sanchez et al., 1997), vectorised greedy selection ──────────────────

def ncn_predict(X_train, y_train, X_test, k=5, pool=None):
    """
    For each query, greedily pick k neighbours whose running centroid
    stays closest to the query, then majority vote.
    pool: restrict candidates to the `pool` nearest points (None = exact).
    """
    preds = np.empty(len(X_test), dtype=y_train.dtype)
    for i, xq in enumerate(X_test):
        if pool is not None and pool < len(X_train):
            idx = np.argpartition(np.linalg.norm(X_train - xq, axis=1), pool - 1)[:pool]
            P, yP = X_train[idx], y_train[idx]
        else:
            P, yP = X_train, y_train

        available = np.ones(len(P), bool)
        running_sum = np.zeros(P.shape[1])
        chosen = []
        for step in range(min(k, len(P))):
            centroids = (running_sum + P) / (step + 1)
            dist = np.linalg.norm(centroids - xq, axis=1)
            dist[~available] = np.inf
            j = int(np.argmin(dist))
            chosen.append(j)
            available[j] = False
            running_sum += P[j]

        labels, counts = np.unique(yP[chosen], return_counts=True)
        preds[i] = labels[np.argmax(counts)]
    return preds


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(X, y, scale, k=5, n_splits=10, n_repeats=10, pool=None):
    """Return fold-level accuracy arrays for KNN and NCN, plus train/test sizes."""
    n_splits = max(2, min(n_splits, np.bincount(y).min()))
    knn_acc, ncn_acc = [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, y[tr])
            knn_acc.append(accuracy_score(y[te], knn.predict(Xte)))
            ncn_acc.append(accuracy_score(y[te], ncn_predict(Xtr, y[tr], Xte, k, pool)))
    n_test = len(y) // n_splits
    return np.array(knn_acc), np.array(ncn_acc), len(y) - n_test, n_test


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold differences."""
    J, mean, var = len(diffs), diffs.mean(), diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = mean / np.sqrt((1 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def scale_heterogeneity(X):
    """log10(max feature std / min feature std) on RAW features. 0 = equal scales."""
    s = X.std(axis=0)
    s = s[s > 0]
    return np.log10(s.max() / s.min())


# ── Main real-dataset analysis ────────────────────────────────────────────────

def run_scaling_analysis(datasets, k=5, n_repeats=10, pool=None):
    rows = []
    for df, target_col, name in datasets:
        X, y = prepare(df, target_col, name)
        row = {"dataset": name, "n": len(y), "d": X.shape[1],
               "scale_heterogeneity": round(scale_heterogeneity(X), 3)}

        for scale in (False, True):
            tag = "scaled" if scale else "unscaled"
            knn, ncn, n_tr, n_te = evaluate(X, y, scale, k, n_repeats=n_repeats, pool=pool)
            t, p = corrected_ttest(ncn - knn, n_tr, n_te)
            row[f"knn_{tag}"] = round(knn.mean(), 4)
            row[f"ncn_{tag}"] = round(ncn.mean(), 4)
            row[f"gap_{tag}"] = round(ncn.mean() - knn.mean(), 4)
            row[f"p_{tag}"] = round(p, 4)

        row["gap_shrinkage"] = round(row["gap_unscaled"] - row["gap_scaled"], 4)
        rows.append(row)
        print(f"    unscaled gap={row['gap_unscaled']:+.4f} (p={row['p_unscaled']:.3f}) | "
              f"scaled gap={row['gap_scaled']:+.4f} (p={row['p_scaled']:.3f})")

    res = pd.DataFrame(rows)
    alpha_bonf = 0.05 / len(res)
    res["sig_unscaled"] = res["p_unscaled"] < alpha_bonf
    res["sig_scaled"] = res["p_scaled"] < alpha_bonf
    res.to_csv(f"{OUT_DIR}/real_scaling_results.csv", index=False)

    print_summary(res, alpha_bonf)
    plot_real(res)
    return res


def print_summary(res, alpha_bonf):
    line = "=" * 100
    print(f"\n{line}\nKNN vs NCN — UNSCALED (original protocol) vs SCALED\n{line}")
    cols = ["dataset", "scale_heterogeneity", "knn_unscaled", "ncn_unscaled", "gap_unscaled",
            "p_unscaled", "knn_scaled", "ncn_scaled", "gap_scaled", "p_scaled", "gap_shrinkage"]
    print(res[cols].to_string(index=False))

    print(f"\nBonferroni threshold ({len(res)} datasets): alpha = {alpha_bonf:.4f}")
    print(f"  Significant NCN-KNN differences, unscaled: {res['sig_unscaled'].sum()}")
    print(f"  Significant NCN-KNN differences, scaled  : {res['sig_scaled'].sum()}")
    print(f"  Mean |gap| unscaled: {res['gap_unscaled'].abs().mean():.4f}")
    print(f"  Mean |gap| scaled  : {res['gap_scaled'].abs().mean():.4f}")

    # Across datasets: is the unscaled gap larger than the scaled gap?
    if (res["gap_shrinkage"] != 0).any():
        w, p = stats.wilcoxon(res["gap_unscaled"], res["gap_scaled"])
        print(f"\nWilcoxon signed-rank (gap unscaled vs scaled): W={w:.1f}, p={p:.4f}")

    # Does scale heterogeneity predict how much the gap shrinks?
    rho, p = stats.spearmanr(res["scale_heterogeneity"], res["gap_shrinkage"])
    r, pr = stats.pearsonr(res["scale_heterogeneity"], res["gap_shrinkage"])
    print(f"Scale heterogeneity vs gap shrinkage: Spearman rho={rho:+.3f} (p={p:.4f}), "
          f"Pearson r={r:+.3f} (p={pr:.4f})")


def plot_real(res):
    # Figure 1: gap per dataset, unscaled vs scaled
    x = np.arange(len(res))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 0.2, res["gap_unscaled"], 0.4, label="Unscaled (original protocol)", color="#D97706")
    ax.bar(x + 0.2, res["gap_scaled"], 0.4, label="Standardised", color="#0891B2")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(res["dataset"], rotation=30, ha="right")
    ax.set_ylabel("NCN − KNN accuracy")
    ax.set_title("NCN advantage with and without feature scaling", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig_real_gap_scaled_vs_unscaled.png", dpi=150)
    plt.close()

    # Figure 2: scale heterogeneity vs gap shrinkage
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(res["scale_heterogeneity"], res["gap_shrinkage"], s=70, color="#7C3AED")
    for _, r in res.iterrows():
        ax.annotate(r["dataset"], (r["scale_heterogeneity"], r["gap_shrinkage"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("Scale heterogeneity  log10(max std / min std)")
    ax.set_ylabel("Gap shrinkage (unscaled gap − scaled gap)")
    ax.set_title("Larger feature-scale differences → larger NCN advantage lost after scaling",
                 fontweight="bold", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig_heterogeneity_vs_shrinkage.png", dpi=150)
    plt.close()


# ── Controlled synthetic evidence ────────────────────────────────────────────

def synthetic_scale_sweep(scales=(1, 2, 5, 10, 20, 50, 100, 300), n_per_class=150,
                          n_repeats=5, k=5, seed=0):
    """
    4 informative features + 1 weakly informative feature whose scale grows.
    Evaluated WITHOUT and WITH scaling. Only the scale of one feature changes.
    """
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0, 1, (n_per_class, 5))
    X1 = rng.normal(0, 1, (n_per_class, 5))
    X1[:, :4] += 0.9
    X1[:, 4] += 0.3
    base = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    rows = []
    print(f"\n{'=' * 70}\nSYNTHETIC: scale of one weak feature increased\n{'=' * 70}")
    print(f"{'scale':>6} {'gap_unscaled':>13} {'gap_scaled':>11}")
    for s in scales:
        X = base.copy()
        X[:, 4] *= s
        row = {"scale": s}
        for scale in (False, True):
            knn, ncn, _, _ = evaluate(X, y, scale, k, n_repeats=n_repeats)
            tag = "scaled" if scale else "unscaled"
            row[f"knn_{tag}"], row[f"ncn_{tag}"] = knn.mean(), ncn.mean()
            row[f"gap_{tag}"] = ncn.mean() - knn.mean()
        rows.append(row)
        print(f"{s:>6} {row['gap_unscaled']:>+13.4f} {row['gap_scaled']:>+11.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/synthetic_scale_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, tag, title in [(axes[0], "unscaled", "Unscaled"), (axes[1], "scaled", "Standardised")]:
        ax.plot(df["scale"], df[f"knn_{tag}"], "o-", label="KNN", color="royalblue")
        ax.plot(df["scale"], df[f"ncn_{tag}"], "s-", label="NCN", color="#D97706")
        ax.set_xscale("log")
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("Scale factor of one weak feature (log)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Controlled experiment: feature-scale heterogeneity drives the NCN−KNN gap",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig_synthetic_scale_sweep.png", dpi=150)
    plt.close()
    return df


def k_sweep(X, y, name, ks=(1, 3, 5, 9, 15), n_repeats=5):
    """At k=1 NCN == KNN by definition; shows how the gap depends on k."""
    rows = []
    print(f"\n{'=' * 70}\nk SWEEP — {name}\n{'=' * 70}")
    print(f"{'k':>3} {'gap_unscaled':>13} {'gap_scaled':>11}")
    for k in ks:
        row = {"k": k}
        for scale in (False, True):
            knn, ncn, _, _ = evaluate(X, y, scale, k, n_repeats=n_repeats)
            row["gap_scaled" if scale else "gap_unscaled"] = ncn.mean() - knn.mean()
        rows.append(row)
        print(f"{k:>3} {row['gap_unscaled']:>+13.4f} {row['gap_scaled']:>+11.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/k_sweep_{name.lower()}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["k"], df["gap_unscaled"], "o-", color="#D97706", label="Unscaled")
    ax.plot(df["k"], df["gap_scaled"], "s-", color="#0891B2", label="Standardised")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("NCN − KNN accuracy")
    ax.set_title(f"{name}: NCN−KNN gap vs k", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig_k_sweep_{name.lower()}.png", dpi=150)
    plt.close()
    return df


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data"
    read = lambda f, **kw: pd.read_csv(os.path.join(base, f), **kw)

    datasets = [
        (read("ecoli.csv"),        "target",                    "Ecoli"),
        (read("glass.csv"),        "target",                    "Glass"),
        (read("hepatitis.csv"),    "target",                    "Hepatitis"),
        (read("ionosphere.csv"),   "target",                    "Ionosphere"),
        (read("iris.csv"),         "target",                    "Iris"),
        (read("lenses.csv"),       "target",                    "Lenses"),
        (read("lung_cancer.csv"),  "target",                    "Lungcancer"),
        (read("soybean.csv"),      "target",                    "Soybean"),
        (read("wine.csv"),         "target",                    "Wine"),
        (read("absenteeism_at_work.csv", sep=";"), "Absenteeism time in hours", "Absence"),
        (read("thyroid.csv"),      "target",                    "Thyroid"),
        (read("cryotherapy.csv"),  "Result_of_Treatment",       "Cryotherapy"),
        (read("stars.csv"),        "Type",                      "Stars"),
        (read("diabetes.csv"),     "Outcome",                   "Diabetes"),
    ]

    # Breast Cancer Wisconsin (569 x 30) matches the original paper's table
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer(as_frame=True).frame
    datasets.append((bc, "target", "BreastCancer"))

    # pool=None -> exact NCN. Set pool=200 if Thyroid is too slow.
    results = run_scaling_analysis(datasets, k=5, n_repeats=10, pool=None)

    synthetic_scale_sweep()

    wine_X, wine_y = prepare(read("wine.csv"), "target", "Wine")
    k_sweep(wine_X, wine_y, "Wine")

    print(f"\nAll outputs saved in ./{OUT_DIR}/")
