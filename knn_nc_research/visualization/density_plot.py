# -*- coding: utf-8 -*-
"""
visualization/density_plot.py
==============================
Smoothed KDE density plot per feature, overlaid across all classes.

Unlike the Histogram class which shows raw bin counts, DensityPlot
shows only the smooth KDE curves — cleaner for papers and reports
where you want to compare distributional shapes without histogram noise.

Usage
-----
    df   = IrisDataset().load()
    plot = DensityPlot(df, target_col="target", title="Iris — Density")
    plot.draw()
    plot.draw(save_path="reports/iris_density.png")
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from visualization.base_plot import BasePlot


class DensityPlot(BasePlot):
    """
    Smoothed KDE density curves per feature, one colour per class.

    Parameters
    ----------
    df         : Full DataFrame including the target column.
    target_col : Name of the class/target column (default "target").
    title      : Figure title (default "").
    bandwidth  : KDE bandwidth method — 'scott', 'silverman', or a float
                 (default 'scott').
    dpi        : Figure DPI (default 130).
    """

    def __init__(self, df, target_col="target", title="",
                 bandwidth="scott", dpi=130):
        super().__init__(df, title, dpi)
        self._target_col = target_col
        self._bandwidth  = bandwidth

    def _build_figure(self):
        feat_cols = [c for c in self._df.columns if c != self._target_col]
        classes   = sorted(self._df[self._target_col].unique())

        n_features = len(feat_cols)
        n_cols     = min(3, n_features)
        n_rows     = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.0 * n_cols, 3.2 * n_rows),
        )
        axes = np.array(axes).flatten() if n_features > 1 else [axes]

        fig.suptitle(
            self._title or "KDE Density Plot per Feature",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for idx, col in enumerate(feat_cols):
            ax = axes[idx]

            # Determine shared x-range across all classes for this feature
            all_vals = self._df[col].dropna()
            x_min    = float(all_vals.min())
            x_max    = float(all_vals.max())
            xs       = np.linspace(x_min, x_max, 400)

            for ci, cls in enumerate(classes):
                vals  = self._df[self._df[self._target_col] == cls][col].dropna().values
                color = self._color(ci)

                if len(vals) < 4:
                    continue

                kde = gaussian_kde(vals, bw_method=self._bandwidth)
                ax.plot(xs, kde(xs), color=color, lw=2.2, label=f"Class {cls}")
                ax.fill_between(xs, kde(xs), alpha=0.12, color=color)

            ax.set_title(col, fontsize=10)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7, loc="upper right")

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig
