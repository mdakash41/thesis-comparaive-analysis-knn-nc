"""
Thesis Table 4.1: size of each dataset as used (after cleaning) and the number
of cross-validation folds, next to the sizes reported in the original paper.
"""

import numpy as np
import pandas as pd

from src.datasets import load_all, n_folds
from src.paths import RESULTS_DIR

# (attributes, objects, classes) from Table I of Mladenova and Valova (2022)
ORIGINAL = {
    "Ecoli": (8, 335, 8), "Glass": (9, 214, 6), "Hepatitis": (19, 155, 2),
    "Ionosphere": (34, 351, 2), "Iris": (4, 150, 3), "Lenses": (4, 24, 3),
    "Lung cancer": (56, 32, 3), "Soybean": (35, 47, 4), "Wine": (13, 178, 3),
    "Absenteeism": (19, 740, 28), "Thyroid": (5, 215, 3), "Cryotherapy": (6, 90, 2),
    "Stars": (4, 240, 6), "Pima diabetes": (8, 768, 2), "Breast cancer": (30, 569, 2),
}


def run(n_repeats=10):
    rows = []
    for name, X, y in load_all():
        folds = n_folds(y)
        d0, n0, c0 = ORIGINAL[name]
        rows.append({"dataset": name, "n": len(y), "d": X.shape[1],
                     "classes": len(np.unique(y)), "smallest_class": np.bincount(y).min(),
                     "folds": folds, "paired_values": folds * n_repeats,
                     "orig_n": n0, "orig_d": d0, "orig_classes": c0})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp0_datasets.csv", index=False)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
