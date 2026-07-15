# -*- coding: utf-8 -*-
"""
visualization/qq_plot.py
=========================
Q-Q (Quantile-Quantile) plots vs the Normal distribution.

One figure per class, one subplot per feature.
Red shading highlights where the sample quantiles deviate from the
normal reference line — direct visual evidence of non-normality.

Usage
-----
    df   = dataset.load()
    plot = QQPlot(df, target_col="target", title="Iris — Q-Q Plots")
    plot.draw()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from visualization.base_plot import BasePlot


class QQPlot(BasePlot):
    """
    Q-Q plots per class and per feature against the normal distribution.

    Parameters
    ----------
    df         : Full DataFrame including the target column.
    target_col : Name of the class/target column (default "target").
    title      : Base title; class label is appended per figure (default "").
    dpi        : Figure DPI (default 130).
    """

    def __init__(self, df, target_col="target", title="", dpi=130):
        super().__init__(df, title, dpi)
        self._target_col = target_col

    def draw(self, save_path=None):
        """
        Render one Q-Q figure per class.

        Parameters
        ----------
        save_path : str or Path prefix. If provided, each figure is saved
                    as <save_path>_class_<label>.png
        """
        classes   = sorted(self._df[self._target_col].unique())
        feat_cols = [c for c in self._df.columns if c != self._target_col]

        for cls in classes:
            subset = self._df[self._df[self._target_col] == cls][feat_cols]
            if len(subset) < 4:
                continue

            fig = self._build_class_figure(subset, cls, feat_cols)

            if save_path is not None:
                from pathlib import Path
                p = Path(str(save_path) + f"_class_{cls}.png")
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p, bbox_inches="tight")

            plt.show()
            plt.close(fig)

    def _build_figure(self):
        # Not used directly — draw() overrides the loop.
        pass

    def _build_class_figure(self, subset, cls, feat_cols):
        n_features = len(feat_cols)
        n_cols     = min(3, n_features)
        n_rows     = int(np.ceil(n_features / n_cols))

        color = self._color(0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.5 * n_rows))
        axes      = np.array(axes).flatten() if n_features > 1 else [axes]

        base_title = self._title or "Q-Q Plots vs Normal Distribution"
        fig.suptitle(
            f"{base_title} | Class: {cls}\n"
            "(Points on dashed line = normal  |  Red shading = deviation)",
            fontsize=12, fontweight="bold", y=1.01,
        )

        for idx, col in enumerate(feat_cols):
            ax   = axes[idx]
            vals = subset[col].dropna().values

            if len(vals) < 4:
                ax.set_visible(False)
                continue

            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            fit_line  = slope * np.array(osm) + intercept
            residuals = osr - fit_line
            std_res   = float(np.std(residuals))

            ax.scatter(osm, osr, color=color, alpha=0.75, s=22, zorder=3)
            ax.plot(osm, fit_line, color="#1E293B", lw=1.5, ls="--",
                    label=f"R²={r**2:.3f}")
            ax.fill_between(osm, fit_line, osr,
                            where=np.abs(residuals) > std_res,
                            alpha=0.20, color="red", label="Deviation")

            ax.set_title(col, fontsize=9)
            ax.set_xlabel("Theoretical Quantiles", fontsize=8)
            ax.set_ylabel("Sample Quantiles", fontsize=8)
            ax.legend(fontsize=8)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig
