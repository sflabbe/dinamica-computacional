r"""Problema 5: Curva de interacción N–M usando una sección discretizada por fibras (malla 2D).

Objetivo
--------
Implementar una definición "tipo fibras" (fiber section) para calcular una curva
de interacción axial–flexión (N–M) con:

* Acero A420 (fy = 420 MPa)
* Hormigón C20/25 (se usa f_ck ≈ 20 MPa como referencia)

La discretización es una **malla 2D (y–z)** de fibras para el hormigón. Para que sea
rápida y robusta, se usa una grilla con distribución *cosine-clustered* (tipo Chebyshev),
que concentra fibras cerca del contorno (refinamiento barato basado en distancia a borde;
en secciones convexas suele dar un efecto similar a un "sizing" guiado por medial axis).

La flexión considerada es uniaxial (N–M), por lo que la ley de deformación sigue siendo
"plane sections remain plane" con \(\varepsilon(y)=\varepsilon_0+\kappa(y_c-y)\).
En ese caso, una malla 2D y una discretización 1D en canto darían resultados casi
equivalentes; aquí se implementa 2D para alinear la definición con literatura/software.

Salida
------
Guarda figuras en ./outputs:

* problem5_fiber_interaction_samples.png
* problem5_fiber_interaction_hull.png
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List, Tuple

# Allow running as:
#   python src/problems/problema5_fiber_section_interaction.py
# without requiring editable installs.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plastic_hinge import (
    ConcreteParabolicRect,
    Fiber2D,
    FiberSection2D,
    NMSurfacePolygon,
    SteelBilinearPerfect,
    rectangular_fiber_mesh,
)


def _outputs_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def mm_to_m(value: float) -> float:
    return float(value) / 1000.0


def cm_to_m(value: float) -> float:
    return float(value) / 100.0


def _rebar_area(phi_m: float) -> float:
    return math.pi * (phi_m / 2.0) ** 2


def build_rc_rect_fiber_section_2d(
    *,
    b: float,
    h: float,
    cover: float,
    fc: float,
    fy: float,
    Es: float = 200e9,
    eps_c0: float = 0.002,
    eps_cu: float = 0.0035,
    ny: int = 60,
    nz: int = 40,
    clustering: str = "cosine",
    # Rebars described as (As_layer, y, n_bars)
    rebar_layers: List[Tuple[float, float, int]] | None = None,
) -> FiberSection2D:
    """Create a rectangular RC fiber section (2D mesh).

    Parameters
    ----------
    b, h : float
        Section width and height [m]. y=0 at top fiber.
    cover : float
        Concrete cover measured from top/bottom to the rebar layer [m].
    fc : float
        Concrete compressive strength [Pa]. Compression positive.
    fy : float
        Steel yield stress [Pa].
    rebar_layers : list of (As, y)
        Steel layer areas [m^2] and depths y from top [m].
    """
    conc = ConcreteParabolicRect(fc=fc, eps_c0=eps_c0, eps_cu=eps_cu)
    steel = SteelBilinearPerfect(fy=fy, Es=Es)

    fibers: List[Fiber2D] = []

    # Concrete: 2D mesh
    fibers.extend(
        rectangular_fiber_mesh(
            b=b,
            h=h,
            ny=int(ny),
            nz=int(nz),
            mat=conc,
            clustering=clustering,
        )
    )

    # Steel: discrete bars spread along the width at each layer
    if rebar_layers is not None:
        for As_layer, y, n_bars in rebar_layers:
            n_bars = max(1, int(n_bars))
            As_bar = float(As_layer) / n_bars
            z_pos = np.linspace(-0.5 * b + cover, 0.5 * b - cover, n_bars)
            for z in z_pos:
                fibers.append(Fiber2D(A=As_bar, y=float(y), z=float(z), mat=steel))

    return FiberSection2D(fibers=fibers, y_c=0.5 * h, z_c=0.0)


def sample_interaction_curve(
    section: FiberSection2D,
    *,
    h: float,
    As_tot: float,
    fy: float,
    eps_cu: float = 0.0035,
    n: int = 120,
    c_min: float = 1e-4,
    c_max: float | None = None,
) -> np.ndarray:
    """Sample N–M points by scanning neutral axis depth c.

    Uses two branches:
    * top compression (kappa > 0, y_na = c)
    * bottom compression (kappa < 0, y_na = h - c)
    """
    if c_max is None:
        c_max = 5.0 * h

    c_vals = np.geomspace(c_min, c_max, n)
    pts: List[Tuple[float, float]] = []

    for c in c_vals:
        # Top in compression
        y_na = c
        kappa = eps_cu / c
        eps0 = kappa * (y_na - section.y_c)
        N, M = section.response(eps0, kappa)
        pts.append((N, M))

        # Bottom in compression
        y_na = h - c
        kappa = -eps_cu / c
        eps0 = kappa * (y_na - section.y_c)
        N, M = section.response(eps0, kappa)
        pts.append((N, M))

    # Pure tension (steel only, concrete tension neglected)
    pts.append((-As_tot * fy, 0.0))

    # Pure compression (uniform strain, kappa = 0)
    N0, _ = section.response(eps_cu, 0.0)
    pts.append((N0, 0.0))

    return np.asarray(pts, dtype=float)


def main() -> None:
    out = _outputs_dir()

    # --- Materials requested by the user ---
    fy = 420e6  # A420 (NCh) [Pa]
    fc = 20e6   # C20/25: use f_ck ≈ 20 MPa as reference [Pa]

    # --- Example section (editable) ---
    # Typical column 40x40 cm, 4Ø16 top + 4Ø16 bottom.
    b = cm_to_m(40.0)
    h = cm_to_m(40.0)
    cover = cm_to_m(4.0)

    phi = mm_to_m(16.0)
    n_bars_layer = 4
    As_layer = float(n_bars_layer) * _rebar_area(phi)
    rebar_layers = [
        (As_layer, cover, n_bars_layer),
        (As_layer, h - cover, n_bars_layer),
    ]
    As_tot = sum(As for As, _, _ in rebar_layers)

    sec = build_rc_rect_fiber_section_2d(
        b=b,
        h=h,
        cover=cover,
        fc=fc,
        fy=fy,
        eps_c0=0.002,
        eps_cu=0.0035,
        ny=70,
        nz=50,
        clustering="cosine",
        rebar_layers=rebar_layers,
    )

    pts = sample_interaction_curve(sec, h=h, As_tot=As_tot, fy=fy, eps_cu=0.0035, n=160)
    pts = pts[np.isfinite(pts).all(axis=1)]
    hull = NMSurfacePolygon.from_points(pts)

    # Plot fiber mesh (quick sanity check)
    # Concrete fibers are many; plot centers with low alpha.
    fig, ax = plt.subplots(figsize=(5, 5))
    ys = np.array([f.y for f in sec.fibers])
    zs = np.array([f.z for f in sec.fibers])
    ax.scatter(zs, ys, s=3, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("y [m] (from top)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "problem5_fiber_mesh.png", dpi=170)
    plt.close(fig)

    # Plot raw samples
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, label="samples")
    ax.set_xlabel("N [N]")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem5_fiber_interaction_samples.png", dpi=170)
    plt.close(fig)

    # Plot convex hull / interaction polygon
    fig, ax = plt.subplots(figsize=(7, 5))
    verts = np.vstack([hull.vertices, hull.vertices[0]])
    ax.plot(verts[:, 0], verts[:, 1], "k-", lw=2.0, label="interaction hull")
    ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.25)
    ax.set_xlabel("N [N]")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem5_fiber_interaction_hull.png", dpi=170)
    plt.close(fig)

    # Small text summary
    (out / "problem5_fiber_interaction_summary.txt").write_text(
        "\n".join(
            [
                "Problem 5: Fiber section N-M interaction",
                f"b={b:.3f} m, h={h:.3f} m, cover={cover:.3f} m",
                f"fc={fc/1e6:.1f} MPa (C20/25 ref), fy={fy/1e6:.1f} MPa (A420)",
                f"As_tot={As_tot*1e6:.1f} mm^2 (two layers, {n_bars_layer} bars each, phi={phi*1e3:.0f} mm)",
                f"n_points={pts.shape[0]}",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
