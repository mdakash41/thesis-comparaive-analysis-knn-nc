# -*- coding: utf-8 -*-
"""
visualization/histogram.py
===========================
Histogram with KDE overlay and Gaussian reference line.

One subplot per numeric feature column. Each class drawn in a
distinct colour. Deviation between the KDE and the dashed Gaussian
reference is visual evidence of non-normality.

Usage
-----
    df     = dataset.load()
    labels = dataset.class_labels()
    plot   = Histogram(df, target_col="target", title="Iris — Histogram")
    plot.draw()
    plot.draw(save_path="reports/iris_histogram.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

from visualization.base_plot import BasePlot


class Histogram(BasePlot):
    """
    Histogram + KDE plot with per-class colouring and Gaussian reference.

    Parameters
    ----------
    df         : Full DataFrame including the target column.
    target_col : Name of the class/target column (default "target").
    title      : Figure title (default "").
    bins       : Number of histogram bins (default 15).
    dpi        : Figure DPI (default 130).
    """

    def __init__(self, df, target_col="target", title="", bins=15, dpi=130):
        super().__init__(df, title, dpi)
        self._target_col = target_col
        self._bins       = bins

    def _build_figure(self):
        feat_cols = [c for c in self._df.columns if c != self._target_col]
        classes   = sorted(self._df[self._target_col].unique())

        n_features = len(feat_cols)
        n_cols     = min(3, n_features)
        n_rows     = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows))
        axes      = np.array(axes).flatten() if n_features > 1 else [axes]

        fig.suptitle(
            self._title or "Histogram + KDE\n(Solid = actual  |  Dashed = Gaussian reference)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for idx, col in enumerate(feat_cols):
            ax = axes[idx]

            for ci, cls in enumerate(classes):
                vals  = self._df[self._df[self._target_col] == cls][col].dropna()
                color = self._color(ci)

                ax.hist(vals, bins=self._bins, alpha=0.28, color=color, density=True)

                if len(vals) > 3:
                    xs = np.linspace(float(vals.min()), float(vals.max()), 300)

                    kde = gaussian_kde(vals)
                    ax.plot(xs, kde(xs), color=color, lw=2.0, label=f"Class {cls}")

                    mu, sigma = float(vals.mean()), float(vals.std())
                    if sigma > 0:
                        ax.plot(xs, stats.norm.pdf(xs, mu, sigma),
                                color=color, lw=1.2, ls="--", alpha=0.65)

            ax.set_title(col, fontsize=10)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig
