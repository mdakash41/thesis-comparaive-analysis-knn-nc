# -*- coding: utf-8 -*-
"""
datasets/hepatitis_dataset.py
UCI ID: 46 | 19 features | 155 instances | 2 classes | 92.9% anomaly rate
High anomaly rate — one of the most extreme class imbalance examples.
"""
from analysis import DatasetAnalyser


hepatitis_analyser = DatasetAnalyser(46, 'hepatitis')
hepatitis_analyser.run_all_tests()