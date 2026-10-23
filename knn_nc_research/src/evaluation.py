"""Repeated stratified cross-validation and the corrected resampled t-test."""

import warnings
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler

from src.classifiers import ncn_predict


def cross_validate(X, y, models=("KNN", "NCN"), scale=True, k=5,
                   n_splits=10, n_repeats=10, ncn_pool=None):
    """
    Fold-level accuracies for each model on identical folds.
    The scaler (if used) is fitted on the training fold only.
    Returns (dict model -> array of accuracies, n_train, n_test).
    """
    n_splits = max(2, min(n_splits, np.bincount(y).min()))
    acc = {m: [] for m in models}
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        with warnings.catch_warnings():
            # Absenteeism has a class with a single member; sklearn warns on every split
            warnings.simplefilter("ignore", UserWarning)
            folds = list(skf.split(X, y))
        for tr, te in folds:
            Xtr, Xte = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            if "KNN" in acc:
                knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, y[tr])
                acc["KNN"].append(accuracy_score(y[te], knn.predict(Xte)))
            if "NCN" in acc:
                pred = ncn_predict(Xtr, y[tr], Xte, k, ncn_pool)
                acc["NCN"].append(accuracy_score(y[te], pred))
            if "NC" in acc:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nc = NearestCentroid().fit(Xtr, y[tr])
                acc["NC"].append(accuracy_score(y[te], nc.predict(Xte)))
    n_test = len(y) // n_splits
    return {m: np.array(v) for m, v in acc.items()}, len(y) - n_test, n_test


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau and Bengio (2003) corrected resampled t-test on paired fold differences."""
    J, mean, var = len(diffs), diffs.mean(), diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = mean / np.sqrt((1 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), J - 1)
