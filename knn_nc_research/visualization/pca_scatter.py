# -*- coding: utf-8 -*-
"""
visualization/pca_scatter.py
=============================
PCA 2D scatter plot — class cluster compactness and centroid positions.

Features are standardised before PCA so that scale differences
do not distort inter-class distances.

Compact, separated clusters with centroids at their geometric centres
are strong visual evidence that NC's single-centroid assumption holds
— and that a KNN vs NC comparison on such data is not neutral.

Usage
-----
    df   = dataset.load()
    plot = PCAScatter(df, target_col="target", title="Iris — PCA 2D")
    plot.draw()
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from visualization.base_plot import BasePlot


class PCAScatter(BasePlot):
    """
    PCA 2D scatter with per-class centroid markers.

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

        if len(feat_cols) < 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "Fewer than 2 features — PCA not applicable.",
                    ha="center", va="center", transform=ax.transAxes)
            return fig

        X = self._df[feat_cols].select_dtypes(include=["number"]).dropna()
        y = self._df.loc[X.index, self._target_col].values

        X_scaled = StandardScaler().fit_transform(X)
        pca      = PCA(n_components=2)
        X_pca    = pca.fit_transform(X_scaled)
        var_exp  = pca.explained_variance_ratio_ * 100

        classes = sorted(set(y))

        fig, ax = plt.subplots(figsize=(7, 5))

        for ci, cls in enumerate(classes):
            mask  = y == cls
            color = self._color(ci)

            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       color=color, label=f"Class {cls}",
                       alpha=0.75, s=38, edgecolors="white", linewidths=0.4)

            cx = float(X_pca[mask, 0].mean())
            cy = float(X_pca[mask, 1].mean())
            ax.scatter(cx, cy, color=color, marker="X", s=200,
                       edgecolors="black", linewidths=1.2, zorder=5)

        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=10)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=10)
        ax.set_title(
            f"{self._title or 'PCA 2D Projection'}\n"
            f"X = class centroid  |  Total variance: {sum(var_exp):.1f}%",
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        return fig
