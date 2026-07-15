# -*- coding: utf-8 -*-
"""
statistical_tests/separability_test.py
========================================
Class Separability Ratio — a custom metric for this research project.

Definition
----------
    Separability Ratio(A, B) = d(μ_A, μ_B) / (σ_A + σ_B)

where:
    d(μ_A, μ_B) = Euclidean distance between class centroids in feature space
    σ_A, σ_B    = mean standard deviation across all features for each class

Interpretation
--------------
    ratio > 1.5  → well-separated: NC trivially produces correct predictions
    ratio > 0.8  → moderate separation: NC likely works but not conclusively
    ratio ≤ 0.8  → poor separation: NC centroid assumption is under stress

A benchmark study where all class pairs have high separability ratios is
biased in favour of NC — the classifiers will agree on most predictions
regardless of their underlying assumptions.

Usage
-----
    full_df = dataset.load()
    test    = SeparabilityTest(full_df, target_col="target")
    result  = test.run_test()
"""

import numpy as np
from itertools import combinations
from statistical_tests.base_test import BaseTest

_HIGH     = 1.5
_MODERATE = 0.8


class SeparabilityTest(BaseTest):
    """
    Pairwise class separability ratio across all class combinations.

    Parameters
    ----------
    df         : Full DataFrame including the target column.
    target_col : Name of the column containing class labels (default "target").
    alpha      : Unused — kept for interface consistency.

    Returns from run_test()
    -----------------------
    dict with keys:
        test_name      : str
        pair_results   : list of dicts (one per class pair)
            class_a               : label
            class_b               : label
            centroid_distance     : float
            combined_spread       : float
            separability_ratio    : float
            verdict               : str  ("HIGH" | "MODERATE" | "LOW")
            interpretation        : str
        avg_ratio      : float
        overall_interpretation : str
    """

    def __init__(self, df, target_col="target", alpha=0.05):
        features = df.drop(columns=[target_col]).select_dtypes(include=["number"])
        super().__init__(features, alpha)
        self._target_col = target_col
        self._labels     = df[target_col].values

    def run_test(self):
        classes = sorted(set(self._labels))

        centroids = {}
        spreads   = {}

        for cls in classes:
            mask          = self._labels == cls
            subset        = self._df.values[mask]
            centroids[cls] = subset.mean(axis=0)
            spreads[cls]   = float(np.std(subset, axis=0, ddof=1).mean())

        pair_results = []

        for cls_a, cls_b in combinations(classes, 2):
            dist  = float(np.linalg.norm(centroids[cls_a] - centroids[cls_b]))
            denom = spreads[cls_a] + spreads[cls_b] + 1e-9
            ratio = dist / denom

            if ratio > _HIGH:
                verdict = "HIGH"
                note    = "Classes well-separated — NC centroid is a reliable discriminator."
            elif ratio > _MODERATE:
                verdict = "MODERATE"
                note    = "Moderate separation — NC likely works but is not conclusive."
            else:
                verdict = "LOW"
                note    = "Poor separation — NC centroid assumption is under stress."

            pair_results.append({
                "class_a":            str(cls_a),
                "class_b":            str(cls_b),
                "centroid_distance":  round(dist, 4),
                "combined_spread":    round(spreads[cls_a] + spreads[cls_b], 4),
                "separability_ratio": round(ratio, 4),
                "verdict":            verdict,
                "interpretation":     f"{cls_a} vs {cls_b}: ratio={ratio:.3f} ({verdict}). {note}",
            })

        avg_ratio = round(
            float(np.mean([r["separability_ratio"] for r in pair_results])), 3
        ) if pair_results else 0.0

        return {
            "test_name":    "Class Separability Ratio",
            "pair_results": pair_results,
            "avg_ratio":    avg_ratio,
            "overall_interpretation": self._interpret(
                avg_ratio > _HIGH,
                f"Average ratio={avg_ratio:.3f} — classes are well-separated; NC is trivially favoured.",
                f"Average ratio={avg_ratio:.3f} — separation is not conclusive; KNN local adaptability matters.",
            ),
        }
