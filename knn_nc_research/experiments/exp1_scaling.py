"""
Thesis Sections 5.1-5.3 (Table 5.1, Figures 5.1-5.2).
KNN vs NCN on raw features (original protocol) and on features standardised
inside each training fold, with the corrected resampled t-test.
"""

import pandas as pd
from scipy import stats

from src.datasets import load_all
from src.evaluation import cross_validate, corrected_ttest
from src.measures import scale_heterogeneity
from src.paths import RESULTS_DIR


def run(k=5, n_repeats=10):
    rows = []
    for name, X, y in load_all():
        row = {"dataset": name, "n": len(y), "d": X.shape[1],
               "heterogeneity": scale_heterogeneity(X)}
        for scale, tag in ((False, "raw"), (True, "std")):
            acc, n_tr, n_te = cross_validate(X, y, ("KNN", "NCN"), scale, k, n_repeats=n_repeats)
            _, p = corrected_ttest(acc["NCN"] - acc["KNN"], n_tr, n_te)
            row[f"knn_{tag}"] = acc["KNN"].mean()
            row[f"ncn_{tag}"] = acc["NCN"].mean()
            row[f"gap_{tag}"] = acc["NCN"].mean() - acc["KNN"].mean()
            row[f"p_{tag}"] = p
        row["shrinkage"] = row["gap_raw"] - row["gap_std"]
        rows.append(row)
        print(f"{name:<14} raw gap {row['gap_raw']:+.4f} (p={row['p_raw']:.3f})  "
              f"std gap {row['gap_std']:+.4f} (p={row['p_std']:.3f})")

    # Round so that exact ties (e.g. Lenses and Lung cancer shrinkage) stay ties
    # in the rank-based tests instead of differing by floating-point noise.
    res = pd.DataFrame(rows).round(6)
    alpha = 0.05 / len(res)
    res["sig_raw"] = res["p_raw"] < alpha
    res["sig_std"] = res["p_std"] < alpha
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "exp1_scaling.csv", index=False)

    print(f"\nBonferroni alpha = {alpha:.4f}; significant raw: {res.sig_raw.sum()}, "
          f"standardised: {res.sig_std.sum()}")
    print(f"Mean |gap| raw {res.gap_raw.abs().mean():.4f}, standardised {res.gap_std.abs().mean():.4f}")
    w, p = stats.wilcoxon(res.gap_raw, res.gap_std)
    print(f"Wilcoxon on gaps: W={w:.1f}, p={p:.4f}")
    w, p = stats.wilcoxon(res.gap_raw.abs(), res.gap_std.abs())
    print(f"Wilcoxon on |gaps|: W={w:.1f}, p={p:.4f}")
    rho, p = stats.spearmanr(res.heterogeneity, res.shrinkage)
    r, pr = stats.pearsonr(res.heterogeneity, res.shrinkage)
    print(f"Heterogeneity vs shrinkage: Spearman {rho:+.3f} (p={p:.4f}), Pearson {r:+.3f} (p={pr:.4f})")
    return res


if __name__ == "__main__":
    run()
