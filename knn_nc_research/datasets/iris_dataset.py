# -*- coding: utf-8 -*-
"""
datasets/iris_dataset.py
UCI ID: 53 | 4 features | 150 instances | 3 classes | Balanced
Classic near-normal benchmark — expected to strongly favour NC.
"""

from analysis import DatasetAnalyser


irish_analyser = DatasetAnalyser(53, 'iris')
irish_analyser.run_all_tests()