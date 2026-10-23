"""
Thesis Section 5.6 (Table 5.2, Figure 5.5).
Skewness is increased while class separation is held fixed.

Each feature of class 0 is log-normal(0, sigma), standardised with its
theoretical mean and std (mean 0, variance 1 at every sigma). Class 1 is the
same distribution shifted by delta on every feature. delta is found by
bisection on a large sample so that the measured separability equals
TARGET_SEP at every sigma. Only the shape (skewness) changes.
"""

import numpy as np
import pandas as pd

from src.evaluation import cross_validate
from src.measures import skewness, separability, compactness
from src.paths import RESULTS_DIR

SIGMAS = (0.1, 0.3, 0.5, 0.8, 1.2, 1.5)
TARGET_SEP = 2.20
N_PER_CLASS = 300
N_FEATURES = 4
N_DATASETS = 5          # independent datasets per sigma
N_REPEATS = 2           # CV repeats per dataset
NCN_POOL = 60           # candidate pool for NCN (approximation, synthetic runs only)


def standard_lognormal(n, d, sigma, rng):
    """Log-normal draws standardised to theoretical mean 0 and variance 1."""
    z = rng.lognormal(0.0, sigma, size=(n, d))
    mean = np.exp(sigma ** 2 / 2)
    std = np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(sigma ** 2))
    return (z - mean) / std


def make_data(sigma, seed, delta, n=N_PER_CLASS):
    rng = np.random.default_rng(seed)
    X0 = standard_lognormal(n, N_FEATURES, sigma, rng)
    X1 = standard_lognormal(n, N_FEATURES, sigma, rng) + delta
    return np.vstack([X0, X1]), np.r_[np.zeros(n, int), np.ones(n, int)]


def calibrate_delta(sigma):
    """Bisection on the shift so that measured separability equals TARGET_SEP."""
    lo, hi = 0.05, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        X, y = make_data(sigma, 12345, mid, n=20000)
        if separability(X, y) < TARGET_SEP:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run():
    rows = []
    for sigma in SIGMAS:
        delta = calibrate_delta(sigma)
        acc = {"KNN": [], "NCN": [], "NC": []}
        meas = []
        for ds in range(N_DATASETS):
            X, y = make_data(sigma, 1000 * ds + int(sigma * 100), delta)
            meas.append((skewness(X, y), separability(X, y), compactness(X, y)))
            fold_acc, _, _ = cross_validate(X, y, ("KNN", "NCN", "NC"), True,
                                            n_repeats=N_REPEATS, ncn_pool=NCN_POOL)
            for m in acc:
                acc[m].extend(fold_acc[m])
        m = np.mean(meas, axis=0)
        row = {"sigma": sigma, "delta": delta, "skewness": m[0],
               "separability": m[1], "compactness": m[2]}
        row.update({k.lower(): np.mean(v) for k, v in acc.items()})
        rows.append(row)
        print(f"sigma {sigma}: skew {m[0]:.2f}  sep {m[1]:.2f}  comp {m[2]:.2f}  "
              f"NCN-KNN {row['ncn'] - row['knn']:+.3f}  NC-KNN {row['nc'] - row['knn']:+.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RESULTS_DIR / "exp4_skewness_fixed_separation.csv", index=False)
    return df


if __name__ == "__main__":
    run()
