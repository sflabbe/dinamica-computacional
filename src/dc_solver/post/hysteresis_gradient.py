"""Hysteresis plotting with a time color gradient.

These helpers are used by Problem 2 and Problem 4 scripts to export
paper-style hysteresis plots where color encodes time/step progression.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1)


def add_time_gradient_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    c: Optional[np.ndarray] = None,
    lw: float = 1.6,
    alpha: float = 1.0,
    cmap: str = "viridis",
    norm: Optional[Normalize] = None,
) -> LineCollection:
    """Add a polyline to ax with a colormap along its length.

    Parameters
    ----------
    ax:
        Target axes.
    x, y:
        1D arrays (same length).
    c:
        Scalar field to map to color (same length). If None, uses step index.
    lw, alpha, cmap:
        Line style.
    norm:
        Optional matplotlib Normalize instance.

    Returns
    -------
    LineCollection
        The added collection (useful for colorbar).
    """
    x = _as_1d(x)
    y = _as_1d(y)
    if x.size < 2:
        raise ValueError("Need at least 2 points for a gradient line.")
    if c is None:
        c = np.arange(x.size, dtype=float)
    c = _as_1d(c)
    if c.size != x.size:
        raise ValueError("c must have same length as x/y.")

    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)  # (n-1, 2, 2)

    if norm is None:
        norm = Normalize(vmin=float(np.min(c)), vmax=float(np.max(c)))

    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])  # n-1 values, one per segment
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def add_colorbar(
    lc: LineCollection,
    ax: plt.Axes,
    *,
    label: str = "time",
) -> None:
    """Attach a colorbar to the axes for a given LineCollection."""
    fig = ax.figure
    fig.colorbar(lc, ax=ax, label=label)
