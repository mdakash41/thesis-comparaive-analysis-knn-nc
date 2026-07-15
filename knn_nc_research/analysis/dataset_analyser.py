# -*- coding: utf-8 -*-
"""
analysis/dataset_analyser.py
"""
from datasets.uci_dataset_loader import UCIDatasetLoader
from statistical_tests import (
    NormalityTest,
    UnimodalityTest,
    SkewnessTest,
    KurtosisTest,
    SeparabilityTest,
)
from visualization import (
    Histogram,
    DensityPlot,
    QQPlot,
    PCAScatter,
    BoxPlot,
)


class DatasetAnalyser:
    """
    Runs all statistical tests and visualisations for any UCIDatasetLoader.

    Parameters
    ----------
    dataset  : any loaded or unloaded UCIDatasetLoader instance.
    alpha    : significance level for all tests (default 0.05).
    save_dir : if provided, plots are saved here as PNG files.
    """

    def __init__(self, uci_id, dataset_name, alpha=0.05, save_dir=None):
        self._dataset  = UCIDatasetLoader(uci_id, dataset_name)
        self._alpha    = alpha
        self._save_dir = save_dir

        if not hasattr(self._dataset, "_df") or self._dataset is None:
            self._dataset.load()

        df         = self._dataset._df
        self._df   = df() if callable(df) else df
        self._name = self._dataset._dataset_name


    # ── Full pipeline ──────────────────────────────────────────────────────────

    def run_full_analysis(self):
        """Run all tests then all plots. Returns dict of test results."""
        print(f"\n{'='*60}")
        print(f"  Full Analysis: {self._name}")
        print(f"{'='*60}")

        test_results = self.run_all_tests()
        self.run_all_plots()

        return {
            "dataset_name": self._name,
            "test_results": test_results,
        }

    # ── All tests ──────────────────────────────────────────────────────────────

    def run_all_tests(self):
        """
        Run all 5 tests. Returns dict with keys:
        normality, unimodality, skewness, kurtosis, separability.
        """
        print(f"\n{'─'*60}")
        print(f"  Statistical Tests — {self._name}")
        print(f"{'─'*60}")

        return {
            "normality":    self.run_normality(),
            "unimodality":  self.run_unimodality(),
            "skewness":     self.run_skewness(),
            "kurtosis":     self.run_kurtosis(),
            "separability": self.run_separability(),
        }

    def run_normality(self):
        """Henze-Zirkler normality test per class. Returns list of result dicts."""
        print(f"\n[Normality — Henze-Zirkler] {self._name}")
        per_class = []

        for cls in self._dataset.class_labels():
            subset = self._dataset.class_subset(cls)
            r      = NormalityTest(subset, alpha=self._alpha).run_test()
            per_class.append({"class": cls, **r})
            print(f"  Class {cls:<15} "
                  f"HZ={r['hz_statistic']}  "
                  f"p={r['p_value']}  "
                  f"Normal={r['is_normal']}")

        return per_class

    def run_unimodality(self):
        """Hartigan Dip Test per feature per class. Returns list of result dicts."""
        print(f"\n[Unimodality — Dip Test] {self._name}")
        per_class = []

        for cls in self._dataset.class_labels():
            subset = self._dataset.class_subset(cls)
            r      = UnimodalityTest(subset, alpha=self._alpha).run_test()
            per_class.append({"class": cls, **r})
            print(f"  Class {cls:<15} "
                  f"{r['pct_unimodal']}% unimodal  "
                  f"→ {r['overall_interpretation']}")

        return per_class

    def run_skewness(self):
        """Skewness per feature per class. Returns list of result dicts."""
        print(f"\n[Skewness] {self._name}")
        per_class = []

        for cls in self._dataset.class_labels():
            subset = self._dataset.class_subset(cls)
            r      = SkewnessTest(subset, alpha=self._alpha).run_test()
            per_class.append({"class": cls, **r})
            print(f"  Class {cls:<15} "
                  f"{r['pct_near_normal']}% near-normal  "
                  f"→ {r['overall_interpretation']}")

        return per_class

    def run_kurtosis(self):
        """Kurtosis per feature per class. Returns list of result dicts."""
        print(f"\n[Kurtosis] {self._name}")
        per_class = []

        for cls in self._dataset.class_labels():
            subset = self._dataset.class_subset(cls)
            r      = KurtosisTest(subset, alpha=self._alpha).run_test()
            per_class.append({"class": cls, **r})
            print(f"  Class {cls:<15} "
                  f"{r['pct_near_normal']}% near-normal  "
                  f"→ {r['overall_interpretation']}")

        return per_class

    def run_separability(self):
        """Class separability ratio across all class pairs. Returns single result dict."""
        print(f"\n[Separability Ratio] {self._name}")
        r = SeparabilityTest(
            self._df,
            target_col=self._dataset.target_column(),
        ).run_test()

        print(f"  Avg ratio: {r['avg_ratio']}")
        for pair in r["pair_results"]:
            print(f"    {pair['interpretation']}")

        return r

    # ── All plots ──────────────────────────────────────────────────────────────

    def run_all_plots(self):
        """Draw all 5 plots: Histogram, Density, Q-Q, PCA, BoxPlot."""
        print(f"\n{'─'*60}")
        print(f"  Visualisations — {self._name}")
        print(f"{'─'*60}")

        self.plot_histogram()
        self.plot_density()
        self.plot_qq()
        self.plot_pca()
        self.plot_boxplot()

    def plot_histogram(self):
        """Histogram + KDE with Gaussian reference overlay."""
        print(f"\n  Histogram + KDE — {self._name}")
        Histogram(
            self._df,
            target_col=self._dataset.target_column(),
            title=f"{self._name} — Histogram + KDE",
        ).draw(save_path=self._save_path("histogram"))

    def plot_density(self):
        """Smooth KDE density curves per class."""
        print(f"\n  Density Plot — {self._name}")
        DensityPlot(
            self._df,
            target_col=self._dataset.target_column(),
            title=f"{self._name} — KDE Density",
        ).draw(save_path=self._save_path("density"))

    def plot_qq(self):
        """Q-Q plots per class per feature."""
        print(f"\n  Q-Q Plots — {self._name}")
        QQPlot(
            self._df,
            target_col=self._dataset.target_column(),
            title=f"{self._name} — Q-Q Plots",
        ).draw(save_path=self._save_path("qq"))

    def plot_pca(self):
        """PCA 2D scatter with centroid markers."""
        print(f"\n  PCA Scatter — {self._name}")
        PCAScatter(
            self._df,
            target_col=self._dataset.target_column(),
            title=f"{self._name} — PCA 2D",
        ).draw(save_path=self._save_path("pca"))

    def plot_boxplot(self):
        """Box plots per feature grouped by class."""
        print(f"\n  Box Plot — {self._name}")
        BoxPlot(
            self._df,
            target_col=self._dataset.target_column(),
            title=f"{self._name} — Box Plots",
        ).draw(save_path=self._save_path("boxplot"))

    # ── Private ────────────────────────────────────────────────────────────────

    def _save_path(self, plot_type):
        if self._save_dir is None:
            return None
        from pathlib import Path
        return Path(self._save_dir) / f"{self._name.replace(' ', '_')}_{plot_type}.png"