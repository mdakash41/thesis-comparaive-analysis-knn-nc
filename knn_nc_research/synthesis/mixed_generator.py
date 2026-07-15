# -*- coding: utf-8 -*-
"""
synthesis/mixed_generator.py
==============================
Generates a dataset that mixes different distribution types across features.

Some features are Gaussian, others log-normal (skewed), others uniform.
This mimics real-world datasets where different measurements have different
statistical properties — testing whether NC can still find a reliable centroid
when its assumptions hold for some features but not others.

Usage
-----
    gen = MixedGenerator(n_samples=600, n_features=6, n_classes=3)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class MixedGenerator(BaseGenerator):
    """
    Dataset with mixed feature distributions: Gaussian + log-normal + uniform.

    Features are divided equally into three groups:
        - First third:  Gaussian (NC-favourable)
        - Middle third: Log-normal/skewed (NC-unfavourable)
        - Last third:   Uniform (NC-neutral)

    Parameters
    ----------
    n_samples   : Total instances (default 600).
    n_features  : Number of features (default 6, minimum 3).
    n_classes   : Number of classes (default 3).
    separation  : Class centre offset (default 4.0).
    noise       : Gaussian noise (default 0.0).
    seed        : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=6, n_classes=3,
                 separation=4.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._separation = separation

    def _generate_data(self):
        counts = self._samples_per_class()
        X_parts, y_parts = [], []

        n1 = self._n_features // 3
        n2 = self._n_features // 3
        n3 = self._n_features - n1 - n2

        for cls in range(self._n_classes):
            n       = counts[cls]
            offset  = cls * self._separation

            gauss   = self._rng.normal(offset, 1.0,   size=(n, n1))
            lognorm = self._rng.lognormal(offset * 0.3, 1.0, size=(n, n2))
            uniform = self._rng.uniform(offset, offset + 4.0, size=(n, n3))

            X_cls   = np.hstack([gauss, lognorm, uniform])
            X_parts.append(X_cls)
            y_parts.append(np.full(n, cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
