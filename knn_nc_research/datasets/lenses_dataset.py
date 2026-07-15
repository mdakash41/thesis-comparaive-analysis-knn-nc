# -*- coding: utf-8 -*-
"""
datasets/lenses_dataset.py
UCI ID: 58 | 4 features | 24 instances | 3 classes | Class imbalance
Smallest dataset in the study — statistical tests must handle small n.
"""
from analysis import DatasetAnalyser


ionosphere_analyser = DatasetAnalyser(58, 'lenses')
ionosphere_analyser.run_all_tests()