# -*- coding: utf-8 -*-
"""
datasets/lung_cancer_dataset.py
UCI ID: 62 | 56 features | 32 instances | 3 classes | 78% anomaly rate
Classic n<<p problem: attributes far outnumber instances.
"""
from analysis import DatasetAnalyser


ionosphere_analyser = DatasetAnalyser(62, 'lung_cancer')
ionosphere_analyser.run_all_tests()