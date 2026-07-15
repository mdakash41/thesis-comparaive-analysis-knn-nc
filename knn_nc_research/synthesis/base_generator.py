# -*- coding: utf-8 -*-
"""
synthesis/base_generator.py
============================
Abstract base for all synthetic dataset generators.

Each generator produces a pandas DataFrame with numeric feature columns
and a "target" column containing integer class labels (0, 1, 2, ...).

Consistent interface
--------------------
    gen = SomeGenerator(n_samples=500, n_features=4, n_classes=3)
    df  = gen.generate()
"""

import abc
import numpy as np
import pandas as pd


class BaseGenerator(abc.ABC):
    """
    Abstract base for synthetic dataset generators.

    Parameters
    ----------
    n_samples  : Total number of instances to generate (default 500).
    n_features : Number of numeric feature columns (default 4).
    n_classes  : Number of class labels (default 2).
    noise      : Standard deviation of Gaussian noise added to features (default 0.0).
    seed       : Random seed for reproducibility (default 42).
    """

    def __init__(self, n_samples=500, n_features=4, n_classes=2, noise=0.0, seed=42):
        self._n_samples  = n_samples
        self._n_features = n_features
        self._n_classes  = n_classes
        self._noise      = noise
        self._seed       = seed
        self._rng        = np.random.default_rng(seed)

    def generate(self):
        """
        Generate and return the synthetic dataset.

        Returns
        -------
        pandas DataFrame with columns:
            feature_0, feature_1, ..., feature_n, target
        """
        X, y = self._generate_data()

        if self._noise > 0:
            X = X + self._rng.normal(0, self._noise, size=X.shape)

        cols = [f"feature_{i}" for i in range(X.shape[1])]
        df   = pd.DataFrame(X, columns=cols)
        df["target"] = y.astype(int)
        return df

    @abc.abstractmethod
    def _generate_data(self):
        """
        Generate raw feature matrix X and label vector y.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """

    def _samples_per_class(self, imbalance_ratio=1.0):
        """
        Compute per-class sample counts.

        Parameters
        ----------
        imbalance_ratio : ratio of majority to minority class size (default 1.0 = balanced).

        Returns
        -------
        list of int, one count per class.
        """
        if self._n_classes == 1:
            return [self._n_samples]

        base  = int(self._n_samples / (self._n_classes - 1 + imbalance_ratio))
        major = int(base * imbalance_ratio)
        rest  = [base] * (self._n_classes - 1)
        counts = [major] + rest
        # Adjust last class to hit exact total
        counts[-1] += self._n_samples - sum(counts)
        return counts
