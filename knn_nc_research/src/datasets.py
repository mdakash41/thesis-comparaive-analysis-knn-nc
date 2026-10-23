"""Loads the 15 benchmark datasets used in the thesis."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder

from src.paths import DATA_DIR

# (file, target column, display name, separator)
DATASETS = [
    ("ecoli.csv", "target", "Ecoli", ","),
    ("glass.csv", "target", "Glass", ","),
    ("hepatitis.csv", "target", "Hepatitis", ","),
    ("ionosphere.csv", "target", "Ionosphere", ","),
    ("iris.csv", "target", "Iris", ","),
    ("lenses.csv", "target", "Lenses", ","),
    ("lung_cancer.csv", "target", "Lung cancer", ","),
    ("soybean.csv", "target", "Soybean", ","),
    ("wine.csv", "target", "Wine", ","),
    ("absenteeism_at_work.csv", "Absenteeism time in hours", "Absenteeism", ";"),
    ("thyroid.csv", "target", "Thyroid", ","),
    ("cryotherapy.csv", "Result_of_Treatment", "Cryotherapy", ","),
    ("stars.csv", "Type", "Stars", ","),
    ("diabetes.csv", "Outcome", "Pima diabetes", ","),
]


def prepare(df, target_col):
    """Drop rows with missing values and non-numeric attributes; encode labels."""
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values.astype(float)
    y = LabelEncoder().fit_transform(df[target_col].values)
    return X, y


def load(name):
    """Return (X, y) for one dataset by display name."""
    if name == "Breast cancer":
        return prepare(load_breast_cancer(as_frame=True).frame, "target")
    for file, target, display, sep in DATASETS:
        if display == name:
            return prepare(pd.read_csv(DATA_DIR / file, sep=sep), target)
    raise ValueError(f"Unknown dataset: {name}")


def load_all():
    """Return a list of (name, X, y) for all 15 datasets."""
    names = [d[2] for d in DATASETS] + ["Breast cancer"]
    return [(name, *load(name)) for name in names]
