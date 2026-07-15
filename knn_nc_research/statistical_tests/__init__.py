# -*- coding: utf-8 -*-
from statistical_tests.base_test        import BaseTest
from statistical_tests.normality_test   import NormalityTest
from statistical_tests.unimodality_test import UnimodalityTest
from statistical_tests.skewness_test    import SkewnessTest
from statistical_tests.kurtosis_test    import KurtosisTest
from statistical_tests.separability_test import SeparabilityTest

__all__ = [
    "BaseTest",
    "NormalityTest",
    "UnimodalityTest",
    "SkewnessTest",
    "KurtosisTest",
    "SeparabilityTest",
]
