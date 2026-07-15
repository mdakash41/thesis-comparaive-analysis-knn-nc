# -*- coding: utf-8 -*-
"""
datasets/soybean_dataset.py
UCI ID: 91 | 35 features | 47 instances | 4 classes | Class imbalance
All features are categorical integers — tests distance metrics on discrete data.
"""
from analysis import DatasetAnalyser


ionosphere_analyser = DatasetAnalyser(91, 'soybean')
ionosphere_analyser.run_all_tests()