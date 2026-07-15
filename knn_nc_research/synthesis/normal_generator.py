# -*- coding: utf-8 -*-
"""
synthesis/normal_generator.py
==============================
Generates classes drawn from well-separated Gaussian (normal) distributions.

This is the NC-ideal scenario: each class is unimodal, roughly spherical,
and well-separated. NC and KNN should agree on nearly all predictions,
making any comparison between them on this data meaningless.

Usage
-----
    gen = NormalGenerator(n_samples=600, n_features=4, n_classes=3, separation=4.0)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class NormalGenerator(BaseGenerator):
    """
    Normal (Gaussian) class distributions with configurable separation.

    Parameters
    ----------
    n_samples   : Total instances (default 600).
    n_features  : Number of features (default 4).
    n_classes   : Number of classes (default 3).
    separation  : Euclidean distance between consecutive class centroids (default 4.0).
    noise       : Gaussian noise added to all features (default 0.0).
    seed        : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=3,
                 separation=4.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._separation = separation

    def _generate_data(self):
        counts = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            centre = np.zeros(self._n_features)
            centre[0] = cls * self._separation
            X_cls = self._rng.normal(centre, 1.0, size=(counts[cls], self._n_features))
            X_parts.append(X_cls)
            y_parts.append(np.full(counts[cls], cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
