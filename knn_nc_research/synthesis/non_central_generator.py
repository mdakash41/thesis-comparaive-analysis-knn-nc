# -*- coding: utf-8 -*-
"""
synthesis/non_central_generator.py
=====================================
Generates classes with no central tendency — scattered, heavily overlapping
distributions where no single point can represent a class well.

Combines three effects that all violate NC's assumptions:
  1. High within-class variance (spread)
  2. Low between-class separation (overlap)
  3. Mild multimodality (two sub-clusters per class)

On this data, NC's centroid falls at the average of a scattered,
overlapping region — providing no useful discriminative information.
KNN's local voting should outperform NC significantly here.

The purpose is to act as a negative control: the extreme case
where NC is expected to fail and KNN is expected to hold.

Usage
-----
    gen = NonCentralGenerator(n_samples=600, n_features=4, n_classes=2)
    df  = gen.generate()
"""

import numpy as np
from synthesis.base_generator import BaseGenerator


class NonCentralGenerator(BaseGenerator):
    """
    Scattered, overlapping, mildly multimodal classes — no central tendency.

    Parameters
    ----------
    n_samples   : Total instances (default 600).
    n_features  : Number of features (default 4).
    n_classes   : Number of classes (default 2).
    spread      : Within-class standard deviation (default 3.5 — very loose).
    separation  : Distance between class centroids (default 1.5 — heavy overlap).
    noise       : Additional Gaussian noise (default 0.2).
    seed        : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=2,
                 spread=3.5, separation=1.5, noise=0.2, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._spread     = spread
        self._separation = separation

    def _generate_data(self):
        counts   = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            n          = counts[cls]
            half       = n // 2
            base_offset = cls * self._separation

            # Sub-cluster 1
            c1     = np.zeros(self._n_features)
            c1[0]  = base_offset
            X_sub1 = self._rng.normal(c1, self._spread, size=(half, self._n_features))

            # Sub-cluster 2 — offset in a different dimension to create mild bimodality
            c2     = np.zeros(self._n_features)
            c2[0]  = base_offset
            c2[1]  = self._spread * 1.5          # shift in second feature
            X_sub2 = self._rng.normal(c2, self._spread,
                                       size=(n - half, self._n_features))

            X_cls  = np.vstack([X_sub1, X_sub2])
            X_parts.append(X_cls)
            y_parts.append(np.full(n, cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
