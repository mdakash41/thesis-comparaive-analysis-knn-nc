# -*- coding: utf-8 -*-
"""
experiments/distribution_sweep.py
===================================
DistributionSweepExperiment runs KNNvsNCExperiment across every
synthetic dataset type in one shot and returns a consolidated
comparison DataFrame.

This is the core experiment for proving the bias hypothesis:
    - On NC-favourable data (normal, central tendency): KNN ≈ NC
    - On NC-unfavourable data (multimodal, skewed, non-central): KNN > NC

Usage
-----
    sweep   = DistributionSweepExperiment(n_samples=600, n_features=4, k=5)
    results = sweep.run()

    print(results["comparison"])       # accuracy/F1 across all scenarios
    print(results["winner_counts"])    # how often each classifier wins
"""

import pandas as pd

from synthesis import (
    CentralTendencyGenerator,
    NormalGenerator,
    UniformGenerator,
    SkewedGenerator,
    HeavyTailGenerator,
    MultimodalGenerator,
    ImbalancedGenerator,
    NonCentralGenerator,
    MixedGenerator,
)
from experiments.knn_vs_nc import KNNvsNCExperiment


# Registry: (label, generator_class, constructor kwargs)
_SCENARIOS = [
    (
        "Central Tendency",
        CentralTendencyGenerator,
        {"spread": 0.3, "separation": 8.0},
    ),
    (
        "Normal (Separated)",
        NormalGenerator,
        {"separation": 4.0},
    ),
    (
        "Uniform",
        UniformGenerator,
        {"class_width": 4.0, "separation": 2.0},
    ),
    (
        "Skewed (Mild)",
        SkewedGenerator,
        {"skewness": 1.0, "separation": 3.0},
    ),
    (
        "Skewed (Strong)",
        SkewedGenerator,
        {"skewness": 2.5, "separation": 3.0},
    ),
    (
        "Heavy-Tailed (df=5)",
        HeavyTailGenerator,
        {"degrees_of_freedom": 5, "separation": 6.0},
    ),
    (
        "Heavy-Tailed (df=2)",
        HeavyTailGenerator,
        {"degrees_of_freedom": 2, "separation": 6.0},
    ),
    (
        "Multimodal (2 modes)",
        MultimodalGenerator,
        {"modes_per_class": 2, "mode_separation": 5.0},
    ),
    (
        "Multimodal (3 modes)",
        MultimodalGenerator,
        {"modes_per_class": 3, "mode_separation": 4.0},
    ),
    (
        "Imbalanced (5:1)",
        ImbalancedGenerator,
        {"imbalance_ratio": 5.0, "separation": 4.0},
    ),
    (
        "Non-Central (Scattered)",
        NonCentralGenerator,
        {"spread": 3.5, "separation": 1.5},
    ),
    (
        "Mixed Features",
        MixedGenerator,
        {"n_features": 6, "separation": 4.0},
    ),
]


class DistributionSweepExperiment:
    """
    Runs KNNvsNCExperiment across all built-in synthetic scenarios.

    Parameters
    ----------
    n_samples  : Instances per dataset (default 600).
    n_features : Feature count (overridden per scenario if scenario sets it).
    n_classes  : Class count for generators (default 2).
    k          : KNN neighbour count (default 5).
    cv_folds   : Cross-validation folds (default 10).
    seed       : Random seed (default 42).
    """

    def __init__(self, n_samples=600, n_features=4, n_classes=2,
                 k=5, cv_folds=10, seed=42):
        self._n_samples  = n_samples
        self._n_features = n_features
        self._n_classes  = n_classes
        self._k          = k
        self._cv_folds   = cv_folds
        self._seed       = seed

    def run(self):
        """
        Execute all scenarios and return consolidated results.

        Returns
        -------
        dict with keys:
            comparison     : DataFrame — one row per scenario, all metrics
            winner_counts  : dict {"KNN": int, "NC": int, "Tie": int}
            scenario_results: list of raw result dicts from KNNvsNCExperiment
        """
        rows             = []
        scenario_results = []

        for label, gen_cls, extra_kwargs in _SCENARIOS:
            print(f"  Running: {label} ...", end=" ", flush=True)

            # Merge base kwargs with scenario-specific ones
            base_kwargs = {
                "n_samples":  self._n_samples,
                "n_features": self._n_features,
                "n_classes":  self._n_classes,
                "seed":       self._seed,
            }
            # Remove n_features from base if scenario overrides it
            if "n_features" in extra_kwargs:
                base_kwargs.pop("n_features")

            gen_kwargs = {**base_kwargs, **extra_kwargs}

            # Some generators don't accept n_classes; handle gracefully
            try:
                gen = gen_cls(**gen_kwargs)
            except TypeError:
                gen_kwargs.pop("n_classes", None)
                gen = gen_cls(**gen_kwargs)

            df     = gen.generate()
            exp    = KNNvsNCExperiment(df, k=self._k, cv_folds=self._cv_folds,
                                       seed=self._seed)
            result = exp.run()
            scenario_results.append({"scenario": label, "result": result})

            knn = result["knn_summary"]
            nc  = result["nc_summary"]
            gap = round(knn["accuracy_mean"] - nc["accuracy_mean"], 4)

            winner = "KNN" if gap > 0.001 else ("NC" if gap < -0.001 else "Tie")
            print(f"KNN={knn['accuracy_mean']:.4f}  NC={nc['accuracy_mean']:.4f}  "
                  f"Gap={gap:+.4f}  → {winner}")

            rows.append({
                "Scenario":       label,
                "KNN Acc":        knn["accuracy_mean"],
                "KNN F1":         knn["f1_mean"],
                "KNN Time (ms)":  knn["exec_time_ms_mean"],
                "NC Acc":         nc["accuracy_mean"],
                "NC F1":          nc["f1_mean"],
                "NC Time (ms)":   nc["exec_time_ms_mean"],
                "Acc Gap (KNN-NC)": gap,
                "Winner":         winner,
            })

        comparison = pd.DataFrame(rows).set_index("Scenario")
        winners    = comparison["Winner"].value_counts().to_dict()

        return {
            "comparison":      comparison,
            "winner_counts":   {"KNN": winners.get("KNN", 0),
                                "NC":  winners.get("NC",  0),
                                "Tie": winners.get("Tie", 0)},
            "scenario_results": scenario_results,
        }
