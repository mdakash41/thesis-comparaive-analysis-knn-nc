# -*- coding: utf-8 -*-
"""
statistical_tests/kurtosis_test.py
====================================
Excess kurtosis analysis — per feature column.

Excess kurtosis measures tail heaviness relative to a Gaussian distribution.
    Gaussian:      excess kurtosis = 0
    Leptokurtic:   excess kurtosis > 0  (heavy tails, sharp peak)
    Platykurtic:   excess kurtosis < 0  (light tails, flat distribution)

Heavy-tailed distributions cause outliers to pull the class centroid
away from the main data mass, degrading NC's reliability.

Usage
-----
    subset = df[df["target"] == "g"].drop(columns=["target"])
    test   = KurtosisTest(subset, threshold=2.0)
    result = test.run_test()
"""

from scipy import stats
from statistical_tests.base_test import BaseTest


class KurtosisTest(BaseTest):
    """
    Per-feature excess kurtosis analysis (Fisher's definition).

    Parameters
    ----------
    df        : DataFrame of numeric features.
    alpha     : unused (kept for interface consistency).
    threshold : |excess kurtosis| below this = near-normal (default 2.0).

    Returns from run_test()
    -----------------------
    dict with keys:
        test_name       : str
        feature_results : list of dicts (one per feature)
            feature          : str
            n                : int
            excess_kurtosis  : float
            tail_type        : str  ("normal" | "heavy" | "light")
            is_near_normal   : bool
            interpretation   : str
        pct_near_normal : float
        overall_interpretation: str
    """

    def __init__(self, df, alpha=0.05, threshold=2.0):
        super().__init__(df, alpha)
        self._threshold = threshold

    def run_test(self):
        feature_results = []

        for col in self._df.columns:
            vals = self._df[col].dropna().values
            n    = len(vals)

            if n < 4:
                feature_results.append({
                    "feature":         col,
                    "n":               n,
                    "excess_kurtosis": None,
                    "tail_type":       "unknown",
                    "is_near_normal":  None,
                    "interpretation":  "Insufficient data.",
                })
                continue

            kurt       = round(float(stats.kurtosis(vals)), 4)   # Fisher (excess)
            near_normal = abs(kurt) < self._threshold

            if kurt > self._threshold:
                tail_type = "heavy"
            elif kurt < -self._threshold:
                tail_type = "light"
            else:
                tail_type = "normal"

            tail_note = {
                "heavy":  "Heavy tails — outliers may distort the centroid.",
                "light":  "Light tails — distribution is more uniform than Gaussian.",
                "normal": "Tail weight is near-Gaussian.",
            }[tail_type]

            feature_results.append({
                "feature":         col,
                "n":               n,
                "excess_kurtosis": kurt,
                "tail_type":       tail_type,
                "is_near_normal":  near_normal,
                "interpretation":  f"{col}: excess kurtosis={kurt:.3f} ({tail_type} tails). {tail_note}",
            })

        definite = [r["is_near_normal"] for r in feature_results if r["is_near_normal"] is not None]
        pct      = (sum(definite) / len(definite) * 100) if definite else 0.0

        return {
            "test_name":              "Excess Kurtosis Analysis",
            "threshold":              self._threshold,
            "feature_results":        feature_results,
            "pct_near_normal":        round(pct, 1),
            "overall_interpretation": self._interpret(
                pct >= 60,
                f"{pct:.0f}% of features have near-normal kurtosis — tail behaviour is NC-favourable.",
                f"Only {pct:.0f}% of features have near-normal kurtosis — heavy/light tails detected.",
            ),
        }
