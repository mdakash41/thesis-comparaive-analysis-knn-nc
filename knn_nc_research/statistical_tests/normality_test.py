# -*- coding: utf-8 -*-
"""
statistical_tests/normality_test.py
=====================================
Henze-Zirkler Multivariate Normality Test.

Tests whether the joint distribution of all numeric features is
multivariate normal. If a class passes this test, the NC centroid
is a theoretically valid and optimal class representative.

Usage
-----
    subset = df[df["target"] == "Iris-setosa"].drop(columns=["target"])
    test   = NormalityTest(subset, alpha=0.05)
    result = test.run_test()
"""

import logging
import pingouin as pg
from statistical_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5


class NormalityTest(BaseTest):
    """
    Henze-Zirkler test for multivariate normality.

    Parameters
    ----------
    df    : DataFrame of numeric features (single class subset recommended).
    alpha : significance level (default 0.05).

    Returns from run_test()
    -----------------------
    dict with keys:
        test_name      : str
        n_samples      : int
        hz_statistic   : float | None
        p_value        : float | None
        is_normal      : bool | None
        interpretation : str
    """

    def run_test(self):
        n = len(self._df)

        if n < _MIN_SAMPLES or self._df.shape[1] < 2:
            return {
                "test_name":     "Henze-Zirkler Multivariate Normality",
                "n_samples":     n,
                "hz_statistic":  None,
                "p_value":       None,
                "is_normal":     None,
                "interpretation": f"Insufficient data (n={n}, need ≥{_MIN_SAMPLES} with ≥2 features).",
            }

        try:
            res = pg.multivariate_normality(self._df, alpha=self._alpha)
            is_normal = bool(res.normal)
            return {
                "test_name":      "Henze-Zirkler Multivariate Normality",
                "n_samples":      n,
                "hz_statistic":   round(float(res.hz), 4),
                "p_value":        round(float(res.pval), 4),
                "is_normal":      is_normal,
                "interpretation": self._interpret(
                    is_normal,
                    "Distribution is multivariate normal — NC centroid is reliable.",
                    "Distribution departs from multivariate normality — NC centroid may be unreliable.",
                ),
            }
        except Exception as exc:
            logger.warning("NormalityTest failed: %s", exc)
            return {
                "test_name":     "Henze-Zirkler Multivariate Normality",
                "n_samples":     n,
                "hz_statistic":  None,
                "p_value":       None,
                "is_normal":     None,
                "interpretation": f"Test could not be computed: {exc}",
            }
