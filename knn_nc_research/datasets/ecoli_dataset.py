# -*- coding: utf-8 -*-
"""
datasets/ecoli_dataset.py
UCI ID: 39 | 8 features | 335 instances | 8 classes | Class imbalance
Highest class count in the study — tests NC under multi-class conditions.
"""
from analysis import DatasetAnalyser


ecoli_analyser = DatasetAnalyser(39, 'ecoli')
ecoli_analyser.run_all_tests()
# ecoli_analyser.run_full_analysis()