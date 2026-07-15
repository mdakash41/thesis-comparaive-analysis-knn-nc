# -*- coding: utf-8 -*-
"""
synthesis/heavy_tail_generator.py
===================================
Generates classes from Student's t distribution (heavy-tailed).

Heavy tails produce extreme outliers that pull the NC centroid away
from the dense region of the class. KNN local voting is more resistant
to this effect because outliers are unlikely to be among the k nearest
neighbours of a typical query point.

Usage
-----
    gen = HeavyTailGenerator(n_samples=600, n_features=4, degrees_of_freedom=2)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class HeavyTailGenerator(BaseGenerator):
    """
    Classes drawn from t-distributions with configurable tail heaviness.

    Parameters
    ----------
    n_samples          : Total instances (default 600).
    n_features         : Number of features (default 4).
    n_classes          : Number of classes (default 2).
    degrees_of_freedom : Controls tail weight — lower df = heavier tails (default 2).
    separation         : Class centre offset (default 6.0).
    noise              : Gaussian noise (default 0.0).
    seed               : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=2,
                 degrees_of_freedom=2, separation=6.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._df_t       = degrees_of_freedom
        self._separation = separation

    def _generate_data(self):
        counts = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            offset = cls * self._separation
            X_cls  = self._rng.standard_t(self._df_t, size=(counts[cls], self._n_features))
            X_cls[:, 0] += offset
            X_parts.append(X_cls)
            y_parts.append(np.full(counts[cls], cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
