# -*- coding: utf-8 -*-
"""
synthesis/skewed_generator.py
===============================
Generates classes from log-normal (skewed) distributions.

Skewed distributions shift the mean away from where most data sits.
NC places its centroid at the mean — so the centroid is no longer
at the densest region of the class. KNN, using local voting, is unaffected.

Usage
-----
    gen = SkewedGenerator(n_samples=600, n_features=4, skewness=2.0)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class SkewedGenerator(BaseGenerator):
    """
    Classes drawn from log-normal distributions with configurable skewness.

    Parameters
    ----------
    n_samples   : Total instances (default 600).
    n_features  : Number of features (default 4).
    n_classes   : Number of classes (default 2).
    skewness    : Log-normal sigma parameter controlling skewness (default 1.5).
                  Higher values = more skewed.
    separation  : Offset between class means (default 3.0).
    noise       : Gaussian noise (default 0.0).
    seed        : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=2,
                 skewness=1.5, separation=3.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._skewness   = skewness
        self._separation = separation

    def _generate_data(self):
        counts = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            mean_offset = cls * self._separation
            X_cls = self._rng.lognormal(
                mean=mean_offset, sigma=self._skewness,
                size=(counts[cls], self._n_features)
            )
            X_parts.append(X_cls)
            y_parts.append(np.full(counts[cls], cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
