# -*- coding: utf-8 -*-
"""
visualization/summary_heatmap.py
=================================
Cross-dataset or cross-scenario summary heatmap and bar chart.

Designed to take the aggregated results from multiple experiments
or analyses and display them in a single comparative figure.

Two panels:
    Left  — Heatmap of percentage-based bias indicators
    Right — Bar chart of a numeric metric (e.g. separability ratio
            or accuracy gap) across datasets/scenarios.

Usage
-----
    # Build a summary DataFrame from your experiment results
    import pandas as pd
    rows = [
        {"Dataset": "Iris",       "KNN Acc": 0.97, "NC Acc": 0.96, "Gap": 0.01},
        {"Dataset": "Ionosphere", "KNN Acc": 0.87, "NC Acc": 0.93, "Gap": -0.06},
    ]
    summary = pd.DataFrame(rows).set_index("Dataset")

    plot = SummaryHeatmap(
        summary,
        heatmap_cols=["KNN Acc", "NC Acc"],
        bar_col="Gap",
        title="KNN vs NC — Accuracy Summary",
    )
    plot.draw()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from visualization.base_plot import BasePlot

_HIGH     = 1.5
_MODERATE = 0.8


class SummaryHeatmap(BasePlot):
    """
    Two-panel summary visualisation: heatmap + bar chart.

    Parameters
    ----------
    summary_df   : pandas DataFrame indexed by dataset/scenario name.
                   Columns contain numeric metrics.
    heatmap_cols : list of column names to show in the heatmap panel.
    bar_col      : column name to use for the bar chart panel.
    title        : overall figure title (default "").
    vmin         : heatmap colour scale minimum (default 0).
    vmax         : heatmap colour scale maximum (default 1).
    cmap         : seaborn/matplotlib colormap for heatmap (default "RdYlGn").
    dpi          : figure DPI (default 130).
    """

    def __init__(self, summary_df, heatmap_cols, bar_col,
                 title="", vmin=0.0, vmax=1.0,
                 cmap="RdYlGn", dpi=130):
        # BasePlot expects a df; pass summary_df
        super().__init__(summary_df, title, dpi)
        self._heatmap_cols = heatmap_cols
        self._bar_col      = bar_col
        self._vmin         = vmin
        self._vmax         = vmax
        self._cmap         = cmap

    def _build_figure(self):
        summary = self._df

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(summary) * 0.6 + 2)))

        # ── Left: Heatmap ─────────────────────────────────────────────────────
        heat_data = summary[self._heatmap_cols].astype(float)

        sns.heatmap(
            heat_data,
            annot=True,
            fmt=".3f",
            cmap=self._cmap,
            vmin=self._vmin,
            vmax=self._vmax,
            ax=axes[0],
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        axes[0].set_title("Metric Comparison Heatmap", fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=30)

        # ── Right: Bar chart ──────────────────────────────────────────────────
        bar_vals = summary[self._bar_col].astype(float)

        bar_colors = [
            "#059669" if v > 0.001
            else "#DC2626" if v < -0.001
            else "#94A3B8"
            for v in bar_vals
        ]

        axes[1].barh(bar_vals.index, bar_vals.values,
                     color=bar_colors, edgecolor="white")
        axes[1].axvline(0, color="#1E293B", lw=1.2, ls="--")
        axes[1].set_title(f"Bar Chart: {self._bar_col}", fontweight="bold")
        axes[1].set_xlabel(self._bar_col)

        # Colour legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#059669", label="KNN better"),
            Patch(facecolor="#DC2626", label="NC better"),
            Patch(facecolor="#94A3B8", label="Tie"),
        ]
        axes[1].legend(handles=legend_elements, fontsize=9, loc="lower right")

        if self._title:
            fig.suptitle(self._title, fontsize=13, fontweight="bold", y=1.02)

        plt.tight_layout()
        return fig
