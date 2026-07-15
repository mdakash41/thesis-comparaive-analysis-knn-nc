# -*- coding: utf-8 -*-
"""
synthesis/multimodal_generator.py
===================================
Generates classes with multiple internal clusters (multimodal distributions).

This violates NC's core assumption. The centroid of a bimodal class falls
in the empty space between the two modes — representing neither.
KNN handles this naturally via local voting; NC will systematically fail.

Usage
-----
    gen = MultimodalGenerator(n_samples=600, n_features=4, modes_per_class=2)
    df  = gen.generate()
"""
import numpy as np
from synthesis.base_generator import BaseGenerator


class MultimodalGenerator(BaseGenerator):
    """
    Each class is composed of multiple Gaussian sub-clusters (modes).

    Parameters
    ----------
    n_samples       : Total instances (default 600).
    n_features      : Number of features (default 4).
    n_classes       : Number of classes (default 2).
    modes_per_class : Number of internal clusters per class (default 2).
    mode_separation : Distance between modes within a class (default 5.0).
    class_separation: Distance between class groups (default 12.0).
    noise           : Gaussian noise (default 0.1).
    seed            : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=2,
                 modes_per_class=2, mode_separation=5.0,
                 class_separation=12.0, noise=0.1, seed=42):
        super().__init__(n_samples, n_features, n_classes, noise, seed)
        self._modes_per_class  = modes_per_class
        self._mode_separation  = mode_separation
        self._class_separation = class_separation

    def _generate_data(self):
        counts   = self._samples_per_class()
        X_parts, y_parts = [], []

        for cls in range(self._n_classes):
            n_cls      = counts[cls]
            per_mode   = n_cls // self._modes_per_class
            class_base = np.zeros(self._n_features)
            class_base[0] = cls * self._class_separation

            for mode in range(self._modes_per_class):
                centre      = class_base.copy()
                centre[1]   = mode * self._mode_separation
                n_this_mode = per_mode if mode < self._modes_per_class - 1 else (n_cls - per_mode * (self._modes_per_class - 1))
                X_mode = self._rng.normal(centre, 0.8, size=(n_this_mode, self._n_features))
                X_parts.append(X_mode)
                y_parts.append(np.full(n_this_mode, cls))

        return np.vstack(X_parts), np.concatenate(y_parts)
