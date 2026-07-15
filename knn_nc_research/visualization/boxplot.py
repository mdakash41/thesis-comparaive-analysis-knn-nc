# -*- coding: utf-8 -*-
"""
visualization/boxplot.py
=========================
Box plot per feature grouped by class.

Box plots reveal median, spread, skewness (asymmetric boxes),
and outliers (points beyond whiskers) simultaneously.
Symmetric boxes with few outliers suggest NC-friendly distributions.

Usage
-----
    df   = dataset.load()
    plot = BoxPlot(df, target_col="target", title="Iris — Box Plots")
    plot.draw()
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.base_plot import BasePlot


class BoxPlot(BasePlot):
    """
    Box-and-whisker plots per feature, grouped by class.

    Parameters
    ----------
    df         : Full DataFrame including the target column.
    target_col : Name of the class/target column (default "target").
    title      : Figure title (default "").
    dpi        : Figure DPI (default 130).
    """

    def __init__(self, df, target_col="target", title="", dpi=130):
        super().__init__(df, title, dpi)
        self._target_col = target_col

    def _build_figure(self):
        feat_cols = [c for c in self._df.columns if c != self._target_col]
        classes   = sorted(self._df[self._target_col].unique())

        n_features = len(feat_cols)
        n_cols     = min(3, n_features)
        n_rows     = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.5 * n_rows))
        axes      = np.array(axes).flatten() if n_features > 1 else [axes]

        fig.suptitle(
            self._title or "Box Plots per Feature by Class",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for idx, col in enumerate(feat_cols):
            ax   = axes[idx]
            data = [
                self._df[self._df[self._target_col] == cls][col].dropna().values
                for cls in classes
            ]
            colors     = [self._color(ci) for ci in range(len(classes))]
            bp         = ax.boxplot(data, patch_artist=True, notch=False,
                                     medianprops={"color": "black", "lw": 2})

            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)

            ax.set_title(col, fontsize=10)
            ax.set_xticks(range(1, len(classes) + 1))
            ax.set_xticklabels([str(c) for c in classes], fontsize=8, rotation=30)
            ax.set_ylabel("Value", fontsize=8)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig
