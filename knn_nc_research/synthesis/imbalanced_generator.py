# -*- coding: utf-8 -*-
"""
synthesis/imbalanced_generator.py
===================================
Generates class-imbalanced datasets based on Gaussian distributions.

Class imbalance means NC and KNN may both struggle with minority classes,
but in different ways. NC's centroid is unaffected by class size — it simply
averages the available points. KNN's majority vote can be dominated by the
larger class near the boundary.

Usage
-----
    gen = ImbalancedGenerator(n_samples=600, n_features=4, imbalance_ratio=5.0)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class ImbalancedGenerator(BaseGenerator):
    """
    Two-class dataset with configurable class imbalance ratio.

    Parameters
    ----------
    n_samples       : Total instances (default 600).
    n_features      : Number of features (default 4).
    imbalance_ratio : Majority:minority class ratio (default 5.0).
    separation      : Class centre offset (default 4.0).
    noise           : Gaussian noise (default 0.0).
    seed            : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, imbalance_ratio=5.0,
                 separation=4.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, 2, noise, seed)
        self._imbalance_ratio = imbalance_ratio
        self._separation      = separation

    def _generate_data(self):
        counts = self._samples_per_class(imbalance_ratio=self._imbalance_ratio)
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            centre    = np.zeros(self._n_features)
            centre[0] = cls * self._separation
            X_cls     = self._rng.normal(centre, 1.0, size=(counts[cls], self._n_features))
            X_parts.append(X_cls)
            y_parts.append(np.full(counts[cls], cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
