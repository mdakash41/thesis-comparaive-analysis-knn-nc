# -*- coding: utf-8 -*-
"""
synthesis/uniform_generator.py
================================
Generates classes from uniform distributions with configurable separation.

Uniform distributions have no central tendency — all values within the
range are equally likely. NC centroid lands at the midpoint of the range,
which is a valid representative only if classes do not overlap.

Usage
-----
    gen = UniformGenerator(n_samples=600, n_features=4, separation=6.0)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class UniformGenerator(BaseGenerator):
    """
    Classes drawn from uniform distributions.

    Parameters
    ----------
    n_samples   : Total instances (default 600).
    n_features  : Number of features (default 4).
    n_classes   : Number of classes (default 2).
    class_width : Width of each class's uniform range (default 4.0).
    separation  : Gap between class ranges (default 2.0).
    noise       : Gaussian noise (default 0.0).
    seed        : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=2,
                 class_width=4.0, separation=2.0, noise=0.0, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._class_width = class_width
        self._separation  = separation

    def _generate_data(self):
        counts = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            low  = cls * (self._class_width + self._separation)
            high = low + self._class_width
            X_cls = self._rng.uniform(low, high, size=(counts[cls], self._n_features))
            X_parts.append(X_cls)
            y_parts.append(np.full(counts[cls], cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
