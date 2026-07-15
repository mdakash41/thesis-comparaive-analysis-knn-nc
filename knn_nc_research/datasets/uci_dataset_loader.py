# -*- coding: utf-8 -*-
"""
datasets/base_dataset.py
========================
Abstract base class for all dataset wrappers.

Every concrete dataset subclass must implement _dataset_name() and
_fetch_from_source(). The base handles all caching, path management,
and data cleaning automatically.

Caching behaviour
-----------------
- First call to load() downloads from UCI and saves a .parquet file
  inside the data/ directory next to the project root.
- Subsequent calls load directly from disk without any network request.

Public interface
----------------
    dataset = IrisDataset()
    df = dataset.load()
    features, target = dataset.features(), dataset.target_column()
"""

import abc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)


class UCIDatasetLoader:
    """
    Abstract base for UCI dataset wrappers with automatic local caching.

    Subclasses declare the dataset identity via abstract methods.
    All fetch, cache, and clean logic lives here.
    """
    def __init__(self, uci_id , dataset_name):
        self._dataset_name = dataset_name
        self._uci_id = uci_id


    def load(self):
        """
        Return the full merged DataFrame (features + target).

        Downloads from UCI on the first call and caches to disk.
        Returns the cached version on all subsequent calls.
        """
        cache_path = _DATA_DIR / f"{self._dataset_name}.csv"

        if cache_path.exists():
            logger.info("Loading %s from cache: %s", self._dataset_name, cache_path)
            df = pd.read_csv(cache_path)
        else:
            logger.info("Fetching %s from UCI ML Repo ...", self._dataset_name)
            df = self._fetch_from_source()
            df = self._clean(df)
            df.to_csv(cache_path, index=False)
            logger.info("Cached %s to %s", self._dataset_name, cache_path)

        self._df = df
        return df

    def features(self):
        """
        Return only the numeric feature columns as a DataFrame.
        Requires load() to have been called first.
        """
        self._require_loaded()
        return self._df.drop(columns=[self._target_col]).select_dtypes(include=[np.number])

    def target_column(self):
        """Return the name of the target/class column."""
        return self._target_col

    def class_labels(self):
        """Return sorted list of unique class labels."""
        self._require_loaded()
        return sorted(self._df[self._target_col].unique().tolist())

    def class_subset(self, label):
        """
        Return feature rows for a single class label.

        Parameters
        ----------
        label : class label value matching entries in the target column.
        """
        self._require_loaded()
        mask = self._df[self._target_col] == label
        return self.features()[mask.values]

    def summary(self):
        """
        Return a dict with basic dataset statistics.

        Keys: name, n_instances, n_features, n_classes, target_col
        """
        self._require_loaded()
        return {
            "name":       self._dataset_name,
            "n_instances": len(self._df),
            "n_features":  self.features().shape[1],
            "n_classes":   len(self.class_labels()),
            "target_col":  self._target_col,
        }

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    def _target_col(self):
        """Name of the target column after merging. Override in subclasses if needed."""
        return "target"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_from_source(self):
        raw      = fetch_ucirepo(id=self._uci_id)
        features = raw.data.features.copy()
        targets  = raw.data.targets.copy()

        features = features.select_dtypes(include=[np.number]).dropna()
        targets  = targets.loc[features.index].copy()
        targets.columns = [self._target_col]

        features = features.reset_index(drop=True)
        targets  = targets.reset_index(drop=True)

        return pd.concat([features, targets], axis=1)

    def _clean(self, df):
        """Remove rows with any remaining NaN. Subclasses may override."""
        return df.dropna().reset_index(drop=True)

    def _require_loaded(self):
        if not hasattr(self, "_df") or self._df is None:
            raise RuntimeError(
                f"Call {self.__class__.__name__}.load() before accessing data."
            )

    def __repr__(self):
        loaded = getattr(self, "_df", None)
        status = f"{len(loaded)}×{loaded.shape[1]}" if loaded is not None else "not loaded"
        return f"<{self.__class__.__name__} [{status}]>"

    def print_title(self):
        print(f"\n{'=' * 65}")
        print(f"Statistical Test on: {self._dataset_name.title()}")
        print(f"{'=' * 65}")