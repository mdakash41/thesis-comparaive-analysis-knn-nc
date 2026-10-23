"""
Thesis Section 5.4 (Figure 5.3).
(a) Synthetic data where only the scale of one weak feature changes.
(b) Wine: NCN - KNN gap as a function of k, raw vs standardised.
"""

import numpy as np
import pandas as pd

from src.datasets import load
from src.evaluation import cross_validate
from src.paths import RESULTS_DIR


def scale_sweep(scales=(1, 2, 5, 10, 20, 50, 100, 300), n_per_class=150, n_repeats=5, k=5):
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1, (n_per_class, 5))
    X1 = rng.normal(0, 1, (n_per_class, 5))
    X1[:, :4] += 0.9          # four informative features
    X1[:, 4] += 0.3           # one weak feature, its scale is varied
    base = np.vstack([X0, X1])
    y = np.r_[np.zeros(n_per_class, int), np.ones(n_per_class, int)]

    rows = []
    for s in scales:
        X = base.copy()
        X[:, 4] *= s
        row = {"scale": s}
        for scale, tag in ((False, "raw"), (True, "std")):
            acc, _, _ = cross_validate(X, y, ("KNN", "NCN"), scale, k, n_repeats=n_repeats)
            row[f"gap_{tag}"] = acc["NCN"].mean() - acc["KNN"].mean()
        rows.append(row)
        print(f"scale {s:>4}: raw {row['gap_raw']:+.4f}  std {row['gap_std']:+.4f}")
    return pd.DataFrame(rows)


def k_sweep(name="Wine", ks=(1, 3, 5, 9, 15), n_repeats=5):
    X, y = load(name)
    rows = []
    for k in ks:
        row = {"k": k}
        for scale, tag in ((False, "raw"), (True, "std")):
            acc, _, _ = cross_validate(X, y, ("KNN", "NCN"), scale, k, n_repeats=n_repeats)
            row[f"gap_{tag}"] = acc["NCN"].mean() - acc["KNN"].mean()
        rows.append(row)
        print(f"{name} k={k:>2}: raw {row['gap_raw']:+.4f}  std {row['gap_std']:+.4f}")
    return pd.DataFrame(rows)


def run():
    RESULTS_DIR.mkdir(exist_ok=True)
    scale_sweep().round(4).to_csv(RESULTS_DIR / "exp2_scale_sweep.csv", index=False)
    k_sweep().round(4).to_csv(RESULTS_DIR / "exp2_k_sweep_wine.csv", index=False)


if __name__ == "__main__":
    run()
