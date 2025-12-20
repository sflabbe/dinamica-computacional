"""Fiber mesh plotting helpers (incl. connectivity / wireframe)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _clustered_edges_01(n: int, clustering: str = "cosine") -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be > 0")
    if n == 1:
        return np.array([0.0, 1.0])
    i = np.arange(n + 1, dtype=float)
    if str(clustering).lower() == "linear":
        return i / n
    # cosine clustering
    return 0.5 * (1.0 - np.cos(np.pi * i / n))


def rect_mesh_centroids(
    b: float,
    h: float,
    ny: int,
    nz: int,
    *,
    y0: float = 0.0,
    z0: float = 0.0,
    clustering: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Y,Z) centroid grids with shape (ny, nz)."""
    y_edges = y0 + float(h) * _clustered_edges_01(int(ny), clustering=clustering)
    z_edges = z0 + float(b) * _clustered_edges_01(int(nz), clustering=clustering)
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    Z, Y = np.meshgrid(z_mid, y_mid)
    return Y, Z


def plot_rect_fiber_mesh_connectivity(
    out_path: Path,
    b: float,
    h: float,
    ny: int,
    nz: int,
    *,
    clustering: str = "cosine",
    y0: float = 0.0,
    z0: float = 0.0,
    title: str = "Fiber mesh connectivity",
    xlabel: str = "z [m]",
    ylabel: str = "y [m]",
    node_size: float = 8.0,
    line_lw: float = 0.6,
) -> None:
    """Plot centroid nodes and draw connectivity lines (wireframe-like)."""
    out_path = Path(out_path)
    Y, Z = rect_mesh_centroids(b=b, h=h, ny=ny, nz=nz, y0=y0, z0=z0, clustering=clustering)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    # lines along z
    for i in range(Y.shape[0]):
        ax.plot(Z[i, :], Y[i, :], "-", lw=line_lw, alpha=0.7)
    # lines along y
    for j in range(Z.shape[1]):
        ax.plot(Z[:, j], Y[:, j], "-", lw=line_lw, alpha=0.7)
    ax.scatter(Z.ravel(), Y.ravel(), s=node_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
