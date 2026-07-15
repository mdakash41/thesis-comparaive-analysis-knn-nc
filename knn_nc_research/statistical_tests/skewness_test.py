# -*- coding: utf-8 -*-
"""
statistical_tests/skewness_test.py
====================================
Skewness analysis — per feature column.

Skewness measures the asymmetry of a distribution around its mean.
A Gaussian distribution has skewness = 0.

When class features are highly skewed, the arithmetic mean (centroid)
is pulled away from the mode — meaning the NC centroid no longer sits
at the densest region of the class, reducing its reliability as a
class representative.

Usage
-----
    subset = df[df["target"] == 1].drop(columns=["target"])
    test   = SkewnessTest(subset, threshold=1.0)
    result = test.run_test()
"""

from scipy import stats
from statistical_tests.base_test import BaseTest


class SkewnessTest(BaseTest):
    """
    Per-feature skewness analysis with configurable near-normal threshold.

    Parameters
    ----------
    df        : DataFrame of numeric features.
    alpha     : unused (kept for interface consistency).
    threshold : |skewness| below this value is considered near-normal (default 1.0).

    Returns from run_test()
    -----------------------
    dict with keys:
        test_name       : str
        feature_results : list of dicts (one per feature)
            feature          : str
            n                : int
            skewness         : float
            is_near_normal   : bool
            interpretation   : str
        pct_near_normal : float
        overall_interpretation: str
    """

    def __init__(self, df, alpha=0.05, threshold=1.0):
        super().__init__(df, alpha)
        self._threshold = threshold

    def run_test(self):
        feature_results = []

        for col in self._df.columns:
            vals = self._df[col].dropna().values
            n    = len(vals)

            if n < 4:
                feature_results.append({
                    "feature":        col,
                    "n":              n,
                    "skewness":       None,
                    "is_near_normal": None,
                    "interpretation": "Insufficient data.",
                })
                continue

            skew        = round(float(stats.skew(vals)), 4)
            near_normal = abs(skew) < self._threshold

            feature_results.append({
                "feature":        col,
                "n":              n,
                "skewness":       skew,
                "is_near_normal": near_normal,
                "interpretation": self._interpret(
                    near_normal,
                    f"{col}: skewness={skew:.3f} — symmetric, mean is a reliable centre.",
                    f"{col}: skewness={skew:.3f} — asymmetric, mean is shifted from the mode.",
                ),
            })

        definite = [r["is_near_normal"] for r in feature_results if r["is_near_normal"] is not None]
        pct      = (sum(definite) / len(definite) * 100) if definite else 0.0

        return {
            "test_name":              "Skewness Analysis",
            "threshold":              self._threshold,
            "feature_results":        feature_results,
            "pct_near_normal":        round(pct, 1),
            "overall_interpretation": self._interpret(
                pct >= 60,
                f"{pct:.0f}% of features have low skewness — data is NC-favourable.",
                f"Only {pct:.0f}% of features have low skewness — asymmetric distributions detected.",
            ),
        }
