# -*- coding: utf-8 -*-
"""
datasets/wine_dataset.py
UCI ID: 109 | 13 features | 178 instances | 3 classes | Balanced
Well-separated classes — high separability ratio, NC-favourable.
"""
from analysis import DatasetAnalyser


ionosphere_analyser = DatasetAnalyser(109, 'wine')
ionosphere_analyser.run_all_tests()