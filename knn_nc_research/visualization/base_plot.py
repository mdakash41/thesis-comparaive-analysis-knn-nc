# -*- coding: utf-8 -*-
"""
visualization/base_plot.py
===========================
Abstract base for all visualization classes.

Every plot class receives its data in the constructor and exposes
a single public method: draw() which renders and optionally saves.

Consistent interface
--------------------
    plot = SomePlot(df, title="My Title")
    plot.draw()
    plot.draw(save_path="output/my_plot.png")
"""

import abc
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_DEFAULT_PALETTE = [
    "#0891B2", "#7C3AED", "#D97706", "#059669",
    "#BE185D", "#1D4ED8", "#B45309", "#047857",
    "#9333EA", "#DC2626",
]


class BasePlot(abc.ABC):
    """
    Abstract base for all visualization classes.

    Parameters
    ----------
    df    : pandas DataFrame to visualize.
    title : plot title (default "").
    dpi   : figure resolution (default 130).
    """

    def __init__(self, df, title="", dpi=130):
        self._df     = df
        self._title  = title
        self._dpi    = dpi
        self._palette = _DEFAULT_PALETTE

    def draw(self, save_path=None):
        """
        Render the plot.

        Parameters
        ----------
        save_path : str or Path, optional.
                    If provided, saves the figure to this path
                    in addition to displaying it.
        """
        plt.rcParams.update({
            "figure.dpi":        self._dpi,
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "font.size":         10,
        })

        fig = self._build_figure()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            logger.info("Saved plot: %s", path)

        plt.show()
        plt.close(fig)

    @abc.abstractmethod
    def _build_figure(self):
        """
        Build and return the matplotlib Figure object.
        Called internally by draw().
        """

    def _color(self, index):
        """Return a palette colour for the given index."""
        return self._palette[index % len(self._palette)]
