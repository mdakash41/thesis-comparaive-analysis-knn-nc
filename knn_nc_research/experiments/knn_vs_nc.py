# -*- coding: utf-8 -*-
"""
experiments/knn_vs_nc.py
==========================
KNN vs Nearest Centroid classification experiment with cross-validation.

Computes accuracy, precision, recall, F1-score, and execution time
for both classifiers on the same dataset under the same conditions.
Returns structured results suitable for comparison tables and analysis.

Usage
-----
    df         = IrisDataset().load()
    experiment = KNNvsNCExperiment(df, k=5, cv_folds=10)
    results    = experiment.run()
    print(results["summary"])
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")


class KNNvsNCExperiment:
    """
    Cross-validated KNN vs Nearest Centroid classification experiment.

    Both classifiers are evaluated under identical conditions:
    same train/test splits, same feature scaling, same metrics.

    Parameters
    ----------
    df          : Full DataFrame including the target column.
    target_col  : Name of the class label column (default "target").
    k           : Number of neighbours for KNN (default 5).
    cv_folds    : Number of cross-validation folds (default 10).
    seed        : Random seed for reproducible fold splits (default 42).
    """

    def __init__(self, df, target_col="target", k=5, cv_folds=10, seed=42):
        self._df         = df.copy()
        self._target_col = target_col
        self._k          = k
        self._cv_folds   = cv_folds
        self._seed       = seed

    def run(self):
        """
        Execute the cross-validation experiment.

        Returns
        -------
        dict with keys:
            knn_fold_scores    : list of per-fold metric dicts for KNN
            nc_fold_scores     : list of per-fold metric dicts for NC
            knn_summary        : dict of mean ± std metrics across folds
            nc_summary         : dict of mean ± std metrics across folds
            comparison_df      : pandas DataFrame comparing both classifiers
            knn_confusion      : aggregated confusion matrix (np.ndarray)
            nc_confusion       : aggregated confusion matrix (np.ndarray)
            dataset_info       : dict with basic dataset statistics
        """
        feat_cols = [c for c in self._df.columns if c != self._target_col]
        X         = self._df[feat_cols].select_dtypes(include=["number"]).dropna().values

        le = LabelEncoder()
        y  = le.fit_transform(
            self._df.loc[self._df[feat_cols[0]].notna(), self._target_col].values
        )

        kfold          = StratifiedKFold(n_splits=self._cv_folds, shuffle=True,
                                          random_state=self._seed)
        knn_scores     = []
        nc_scores      = []
        knn_cm_total   = None
        nc_cm_total    = None

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            knn_result   = self._evaluate(KNeighborsClassifier(n_neighbors=self._k),
                                          X_train, y_train, X_test, y_test, fold)
            nc_result    = self._evaluate(NearestCentroid(),
                                          X_train, y_train, X_test, y_test, fold)

            knn_scores.append(knn_result)
            nc_scores.append(nc_result)

            knn_cm_total = (knn_result["confusion_matrix"] if knn_cm_total is None
                            else knn_cm_total + knn_result["confusion_matrix"])
            nc_cm_total  = (nc_result["confusion_matrix"] if nc_cm_total is None
                            else nc_cm_total + nc_result["confusion_matrix"])

        knn_summary = self._summarise(knn_scores, "KNN")
        nc_summary  = self._summarise(nc_scores,  "NC")

        return {
            "knn_fold_scores": knn_scores,
            "nc_fold_scores":  nc_scores,
            "knn_summary":     knn_summary,
            "nc_summary":      nc_summary,
            "comparison_df":   self._comparison_table(knn_summary, nc_summary),
            "knn_confusion":   knn_cm_total,
            "nc_confusion":    nc_cm_total,
            "dataset_info": {
                "n_instances":  len(X),
                "n_features":   X.shape[1],
                "n_classes":    len(le.classes_),
                "cv_folds":     self._cv_folds,
                "k":            self._k,
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _evaluate(self, clf, X_train, y_train, X_test, y_test, fold):
        """Fit classifier, predict, and collect metrics for one fold."""
        t_start = time.perf_counter()

        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        except Exception as exc:
            return {
                "fold": fold, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "exec_time_ms": 0.0,
                "confusion_matrix": np.zeros((2, 2), dtype=int),
                "error": str(exc),
            }

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        avg        = "weighted"

        return {
            "fold":             fold,
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "precision":        round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "exec_time_ms":     round(elapsed_ms, 3),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    def _summarise(self, fold_scores, clf_name):
        """Compute mean ± std across all folds."""
        metrics = ["accuracy", "precision", "recall", "f1", "exec_time_ms"]
        summary = {"classifier": clf_name}

        for m in metrics:
            vals        = [s[m] for s in fold_scores]
            summary[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{m}_std"]  = round(float(np.std(vals)), 4)

        return summary

    def _comparison_table(self, knn_summary, nc_summary):
        """Build a tidy comparison DataFrame."""
        metrics = ["accuracy", "precision", "recall", "f1", "exec_time_ms"]
        rows    = []

        for m in metrics:
            knn_val = knn_summary[f"{m}_mean"]
            nc_val  = nc_summary[f"{m}_mean"]
            diff    = round(knn_val - nc_val, 4)

            rows.append({
                "Metric":     m,
                "KNN (mean)": knn_val,
                "KNN (std)":  knn_summary[f"{m}_std"],
                "NC (mean)":  nc_val,
                "NC (std)":   nc_summary[f"{m}_std"],
                "KNN - NC":   diff,
                "Winner":     "KNN" if diff > 0.001 else ("NC" if diff < -0.001 else "Tie"),
            })

        return pd.DataFrame(rows).set_index("Metric")
