# -*- coding: utf-8 -*-
"""
datasets/glass_dataset.py
UCI ID: 42 | 9 features | 214 instances | 6 classes | Class imbalance
Overlapping chemical composition features across 6 glass types.
"""
from analysis import DatasetAnalyser


glass_analyser = DatasetAnalyser(42, 'glass')
glass_analyser.run_all_tests()