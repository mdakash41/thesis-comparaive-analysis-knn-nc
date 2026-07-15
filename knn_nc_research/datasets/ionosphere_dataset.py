# -*- coding: utf-8 -*-
"""
datasets/ionosphere_dataset.py
UCI ID: 52 | 34 features | 351 instances | 2 classes | Class imbalance
KEY dataset: NC showed statistically significant improvement over KNN
in Mladenova & Valova (2022). High-dimensional.
"""
from analysis import DatasetAnalyser


ionosphere_analyser = DatasetAnalyser(52, 'ionosphere')
ionosphere_analyser.run_all_tests()