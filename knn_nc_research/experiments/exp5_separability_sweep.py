"""
Thesis Section 5.7 (Table 5.3).
Two symmetric Gaussian classes; the distance between their centres is varied.
"""

import numpy as np
import pandas as pd

from src.evaluation import cross_validate
from src.measures import separability
from src.paths import RESULTS_DIR

DISTANCES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)


def run(n_per_class=300, n_features=4, n_datasets=5):
    rows = []
    for dist in DISTANCES:
        acc = {"KNN": [], "NCN": [], "NC": []}
        seps = []
        for ds in range(n_datasets):
            rng = np.random.default_rng(ds)
            shift = dist / np.sqrt(n_features)    # Euclidean distance between centres = dist
            X = np.vstack([rng.normal(0, 1, (n_per_class, n_features)),
                           rng.normal(shift, 1, (n_per_class, n_features))])
            y = np.r_[np.zeros(n_per_class, int), np.ones(n_per_class, int)]
            seps.append(separability(X, y))
            fold_acc, _, _ = cross_validate(X, y, ("KNN", "NCN", "NC"), True,
                                            n_repeats=1, ncn_pool=60)
            for m in acc:
                acc[m].extend(fold_acc[m])
        row = {"distance": dist, "separability": np.mean(seps),
               **{k.lower(): np.mean(v) for k, v in acc.items()}}
        rows.append(row)
        print(f"distance {dist}: sep {row['separability']:.2f}  "
              f"NCN-KNN {row['ncn'] - row['knn']:+.3f}  NC-KNN {row['nc'] - row['knn']:+.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RESULTS_DIR / "exp5_separability_sweep.csv", index=False)
    return df


if __name__ == "__main__":
    run()
