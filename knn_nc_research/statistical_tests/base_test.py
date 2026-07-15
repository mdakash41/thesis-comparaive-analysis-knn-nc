# -*- coding: utf-8 -*-
"""
statistical_tests/base_test.py
===============================
Abstract base for all statistical test classes.

Every test receives a DataFrame in its constructor and exposes
a single public method: run_test() which returns a structured dict.

Consistent interface
--------------------
    test = SomeTest(df, alpha=0.05)
    result = test.run_test()

Result dict always contains at minimum:
    test_name     : str
    interpretation: str
"""

import abc
import pandas as pd


class BaseTest(abc.ABC):
    """
    Abstract base for all statistical tests.

    Parameters
    ----------
    df    : pandas DataFrame containing the data to test.
            Must already be filtered to the relevant class subset
            or full dataset as required by the caller.
    alpha : significance level used for pass/fail decisions (default 0.05).
    """

    def __init__(self, df, alpha=0.05):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        self._df    = df.select_dtypes(include=["number"]).dropna()
        self._alpha = alpha

    @abc.abstractmethod
    def run_test(self):
        """
        Execute the statistical test.

        Returns
        -------
        dict
            Structured result with at minimum:
                test_name      : str
                interpretation : str
        """

    def _interpret(self, passed, pass_msg, fail_msg):
        """Helper: return the appropriate interpretation string."""
        return pass_msg if passed else fail_msg
