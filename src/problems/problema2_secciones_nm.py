"""Problema 2: Secciones RC y curvas de interacción N-M."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plastic_hinge import RCSectionRect, RebarLayer, NMSurfacePolygon


def tonf_cm2_to_Pa(value: float) -> float:
    """Convert tonf/cm^2 to Pa."""
    return float(value) * 9.80665e7


def cm_to_m(value: float) -> float:
    """Convert cm to meters."""
    return float(value) / 100.0


def mm_to_m(value: float) -> float:
    """Convert mm to meters."""
    return float(value) / 1000.0


def _rebar_area(phi_m: float) -> float:
    return math.pi * (phi_m / 2.0) ** 2


def make_section_S1(
    fc: float = 30e6,
    fy: float = 420e6,
    Es: float = 200e9,
    cover: float = 0.05,
    n_fibers: int = 200,
) -> RCSectionRect:
    """Build RC section S1 (column)."""
    b = cm_to_m(40.0)
    h = cm_to_m(60.0)
    As_layer = 4.0 * _rebar_area(mm_to_m(20.0))
    layers = [
        RebarLayer(As=As_layer, y=cover),
        RebarLayer(As=As_layer, y=h - cover),
    ]
    return RCSectionRect(b=b, h=h, fc=fc, fy=fy, Es=Es, layers=layers, n_fibers=n_fibers)


def make_section_S2(
    fc: float = 30e6,
    fy: float = 420e6,
    Es: float = 200e9,
    cover: float = 0.05,
    n_fibers: int = 200,
) -> RCSectionRect:
    """Build RC section S2 (beam)."""
    b = cm_to_m(25.0)
    h = cm_to_m(50.0)
    As_top = 3.0 * _rebar_area(mm_to_m(20.0))
    As_bot = 2.0 * _rebar_area(mm_to_m(20.0))
    layers = [
        RebarLayer(As=As_top, y=cover),
        RebarLayer(As=As_bot, y=h - cover),
    ]
    return RCSectionRect(b=b, h=h, fc=fc, fy=fy, Es=Es, layers=layers, n_fibers=n_fibers)


def compute_interaction_polygon(
    section: RCSectionRect,
    symmetric_M: bool = True,
    symmetric_N: bool = False,
    n: int = 80,
) -> NMSurfacePolygon:
    """Compute convex polygon from sampled N-M points."""
    pts = section.sample_interaction_curve(n=n)
    samples = [pts]
    if symmetric_M:
        samples.append(pts * np.array([1.0, -1.0]))
    if symmetric_N:
        samples.append(pts * np.array([-1.0, 1.0]))
    if symmetric_M and symmetric_N:
        samples.append(pts * np.array([-1.0, -1.0]))
    all_pts = np.vstack(samples)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    return NMSurfacePolygon.from_points(all_pts)


def _outputs_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_nm(ax: plt.Axes, points: np.ndarray, poly: NMSurfacePolygon, label: str) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=14, alpha=0.6, label=f"{label} samples")
    verts = np.vstack([poly.vertices, poly.vertices[0]])
    ax.plot(verts[:, 0], verts[:, 1], lw=2.0, label=f"{label} hull")
    ax.set_xlabel("N [N]")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)


def plot_interaction_curves(n: int = 80) -> Tuple[NMSurfacePolygon, NMSurfacePolygon]:
    """Generate interaction plots for S1 and S2."""
    out = _outputs_dir()
    sec_s1 = make_section_S1()
    sec_s2 = make_section_S2()

    pts_s1 = sec_s1.sample_interaction_curve(n=n)
    poly_s1 = compute_interaction_polygon(sec_s1, symmetric_M=True, symmetric_N=False, n=n)

    pts_s2 = sec_s2.sample_interaction_curve(n=n)
    poly_s2 = compute_interaction_polygon(sec_s2, symmetric_M=True, symmetric_N=False, n=n)

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_nm(ax, pts_s1, poly_s1, "S1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem2_nm_S1.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_nm(ax, pts_s2, poly_s2, "S2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem2_nm_S2.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_nm(ax, pts_s1, poly_s1, "S1")
    _plot_nm(ax, pts_s2, poly_s2, "S2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem2_nm_compare.png", dpi=160)
    plt.close(fig)

    return poly_s1, poly_s2


def main() -> None:
    plot_interaction_curves()


if __name__ == "__main__":
    main()
