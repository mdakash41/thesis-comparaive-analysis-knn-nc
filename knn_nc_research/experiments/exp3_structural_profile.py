"""
Thesis Table 4.1 and Section 5.5 (Figure 5.4).
Structural profile of every dataset and a three-way comparison
(KNN, NCN, NC) on standardised features.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.datasets import load_all
from src.evaluation import cross_validate
from src.measures import profile
from src.paths import RESULTS_DIR

MEASURES = ["skewness", "kurtosis", "separability", "compactness", "modality"]


def run(k=5, n_repeats=10):
    rows = []
    for name, X, y in load_all():
        acc, _, _ = cross_validate(X, y, ("KNN", "NCN", "NC"), True, k, n_repeats=n_repeats)
        row = {"dataset": name, **profile(X, y),
               **{m.lower(): a.mean() for m, a in acc.items()}}
        row["gap_ncn"] = row["ncn"] - row["knn"]
        row["gap_nc"] = row["nc"] - row["knn"]
        rows.append(row)
        print(f"{name:<14} KNN {row['knn']:.4f}  NCN {row['ncn']:.4f}  NC {row['nc']:.4f}")

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.round(4).to_csv(RESULTS_DIR / "exp3_structural_profile.csv", index=False)

    print(f"\nNCN > KNN on {(res.gap_ncn > 1e-4).sum()}/{len(res)}, mean {res.gap_ncn.mean():+.4f}")
    print(f"NC  > KNN on {(res.gap_nc > 1e-4).sum()}/{len(res)}, mean {res.gap_nc.mean():+.4f}")
    print(f"\nCorrelation with the gap (Bonferroni alpha = {0.05 / len(MEASURES):.3f})")
    for m in MEASURES:
        d = res.dropna(subset=[m])
        for gap in ("gap_ncn", "gap_nc"):
            r, p = stats.pearsonr(d[m], d[gap])
            rho, ps = stats.spearmanr(d[m], d[gap])
            print(f"  {m:<13} {gap:<8} Pearson {r:+.3f} (p={p:.3f})  Spearman {rho:+.3f} (p={ps:.3f})  n={len(d)}")

    # Leave-one-out check for the skewness result (Thyroid is the most skewed dataset)
    d = res[res.dataset != "Thyroid"].dropna(subset=["skewness"])
    r, p = stats.pearsonr(d.skewness, d.gap_nc)
    print(f"\nSkewness vs NC gap without Thyroid: r={r:+.3f} (p={p:.3f})")
    return res


if __name__ == "__main__":
    run()
