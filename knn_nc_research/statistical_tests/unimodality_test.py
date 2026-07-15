# -*- coding: utf-8 -*-
"""
statistical_tests/unimodality_test.py
======================================
Hartigan Dip Test for unimodality — applied per feature column.

A class that is unimodal in every feature has a single cluster around
one centre point, making the NC centroid a valid representative.
Multimodal classes break this assumption — the centroid falls in the
empty space between modes and NC will systematically misclassify.

Usage
-----
    subset = df[df["target"] == "g"].drop(columns=["target"])
    test   = UnimodalityTest(subset, alpha=0.05)
    result = test.run_test()
"""

import diptest
from statistical_tests.base_test import BaseTest

_MIN_SAMPLES = 5


class UnimodalityTest(BaseTest):
    """
    Hartigan Dip Test for unimodality, applied to each feature independently.

    H0: distribution is unimodal.
    p-value > alpha  → fail to reject H0 → unimodal (NC-favourable).
    p-value ≤ alpha  → reject H0         → multimodal (NC assumption violated).

    Parameters
    ----------
    df    : DataFrame of numeric features.
    alpha : significance level (default 0.05).

    Returns from run_test()
    -----------------------
    dict with keys:
        test_name        : str
        feature_results  : list of dicts (one per feature)
            feature      : str
            n            : int
            dip_statistic: float
            p_value      : float
            is_unimodal  : bool
            interpretation: str
        pct_unimodal     : float   (% of features that are unimodal)
        overall_interpretation : str
    """

    def run_test(self):
        feature_results = []

        for col in self._df.columns:
            vals = self._df[col].dropna().values
            n    = len(vals)

            if n < _MIN_SAMPLES:
                feature_results.append({
                    "feature":       col,
                    "n":             n,
                    "dip_statistic": None,
                    "p_value":       None,
                    "is_unimodal":   None,
                    "interpretation": f"Insufficient data (n={n}).",
                })
                continue

            dip_stat, pval = diptest.diptest(vals)
            is_unimodal    = pval > self._alpha

            feature_results.append({
                "feature":       col,
                "n":             n,
                "dip_statistic": round(float(dip_stat), 5),
                "p_value":       round(float(pval), 4),
                "is_unimodal":   is_unimodal,
                "interpretation": self._interpret(
                    is_unimodal,
                    f"{col} is unimodal — single cluster, centroid is valid.",
                    f"{col} is multimodal — NC centroid may fall in empty region.",
                ),
            })

        definite    = [r["is_unimodal"] for r in feature_results if r["is_unimodal"] is not None]
        pct_unimodal = (sum(definite) / len(definite) * 100) if definite else 0.0

        return {
            "test_name":              "Hartigan Dip Test (Unimodality)",
            "feature_results":        feature_results,
            "pct_unimodal":           round(pct_unimodal, 1),
            "overall_interpretation": self._interpret(
                pct_unimodal >= 70,
                f"{pct_unimodal:.0f}% of features are unimodal — NC centroid is structurally valid.",
                f"Only {pct_unimodal:.0f}% of features are unimodal — multimodal structure detected.",
            ),
        }
