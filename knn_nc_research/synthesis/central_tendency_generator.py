# -*- coding: utf-8 -*-
"""
synthesis/central_tendency_generator.py
========================================
Generates classes with very strong central tendency — tight Gaussian
clusters with small variance and large between-class separation.

This is the most NC-favourable scenario possible.
Every class has one clear centre, very low spread, and no overlap.
The centroid is an almost perfect class representative.

The purpose is to act as a positive control: on this data,
NC should match or beat KNN in every metric, confirming that
NC works well *when its assumptions are met*.

Usage
-----
    gen = CentralTendencyGenerator(n_samples=600, n_features=4,
                                   n_classes=3, spread=0.3, separation=8.0)
    df  = gen.generate()
"""

import numpy as np
from synthesis.base_generator import BaseGenerator


class CentralTendencyGenerator(BaseGenerator):
    """
    Tightly clustered Gaussian classes — maximum central tendency.

    Parameters
    ----------
    n_samples   : Total instances (default 600).
    n_features  : Number of features (default 4).
    n_classes   : Number of classes (default 3).
    spread      : Within-class standard deviation (default 0.3 — very tight).
    separation  : Distance between consecutive class centroids (default 8.0).
    noise       : Additional Gaussian noise (default 0.0).
    seed        : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=3,
                 spread=0.3, separation=8.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._spread     = spread
        self._separation = separation

    def _generate_data(self):
        counts   = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            centre    = np.zeros(self._n_features)
            centre[0] = cls * self._separation
            X_cls     = self._rng.normal(centre, self._spread,
                                         size=(counts[cls], self._n_features))
            X_parts.append(X_cls)
            y_parts.append(np.full(counts[cls], cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
