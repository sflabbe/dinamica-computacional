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


import argparse
import csv

from plastic_hinge import FiberSection2DStateful
from dc_solver.hinges.models import FiberBeamHinge1D
from dc_solver.post.hysteresis_gradient import add_time_gradient_line, add_colorbar


DEFAULT_STEPS_PER_HALF = 125  # ~501 points per cycle (4*steps_per_half+1)


def _cycle_path(amplitude: float, steps_per_half: int = DEFAULT_STEPS_PER_HALF) -> np.ndarray:
    up = np.linspace(0.0, amplitude, steps_per_half, endpoint=False)
    down = np.linspace(amplitude, 0.0, steps_per_half, endpoint=False)
    neg = np.linspace(0.0, -amplitude, steps_per_half, endpoint=False)
    back = np.linspace(-amplitude, 0.0, steps_per_half + 1)
    return np.concatenate([up, down, neg, back])


def _history_from_cycles(amplitudes, steps_per_half: int = DEFAULT_STEPS_PER_HALF) -> np.ndarray:
    series = []
    for amp in amplitudes:
        cyc = _cycle_path(float(amp), steps_per_half=steps_per_half)
        if series:
            cyc = cyc[1:]
        series.extend(cyc.tolist())
    return np.asarray(series, dtype=float)


def _to_stateful(section: FiberSection2D) -> FiberSection2DStateful:
    # Reuse the same fibers definition (materials must be ConcreteParabolicRect + SteelBilinearPerfect for JIT).
    return FiberSection2DStateful(fibers=section.fibers, y_c=section.y_c, z_c=section.z_c)


def _write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def run_fiber_tests(
    *,
    sec_beam: FiberSection2D,
    sec_col: FiberSection2D,
    h_beam: float,
    h_col: float,
    out: Path,
    steps_per_half: int = DEFAULT_STEPS_PER_HALF,
) -> None:
    """Run 'Problem2/3-style' cyclic tests for fiber hinges/sections.

    Outputs (in ./outputs):
    - problem5_fiber_beam_mtheta_gradient.png + CSV
    - problem5_fiber_col_Nu_gradient.png + CSV
    - problem5_fiber_col_NM_gradient.png + CSV
    """

    # --- Beam: M-theta using a stateful fiber hinge (N_target ~ 0) ---
    beam_state = _to_stateful(sec_beam)
    Lp_beam = 0.5 * float(h_beam)
    hinge = FiberBeamHinge1D(section=beam_state, Lp=Lp_beam, N_target=0.0)

    theta = _history_from_cycles([0.002, 0.004, 0.006, 0.008], steps_per_half=steps_per_half)
    dtheta = np.diff(theta, prepend=theta[0])

    M = np.zeros_like(theta)
    N_res = np.zeros_like(theta)
    eps0 = np.zeros_like(theta)
    iters = np.zeros_like(theta, dtype=int)

    hinge.reset_state()
    for i in range(1, theta.size):
        _M, _k, _th, _a, _M2, extra = hinge.eval_increment(float(dtheta[i]))
        hinge.commit()
        M[i] = float(_M2)
        N_res[i] = float(extra.get("N_res", 0.0))
        eps0[i] = float(extra.get("eps0", 0.0))
        iters[i] = int(extra.get("iters", 0))

    # Plot M-theta (time gradient)
    fig, ax = plt.subplots(figsize=(7, 5))
    lc = add_time_gradient_line(ax, theta, M, c=np.arange(theta.size))
    add_colorbar(lc, ax, label="step")
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "problem5_fiber_beam_mtheta_gradient.png", dpi=180)
    plt.close(fig)

    _write_csv(
        out / "problem5_fiber_beam_mtheta.csv",
        header=["step", "theta_rad", "M_Nm", "N_res_N", "eps0", "iters"],
        rows=[(i, float(theta[i]), float(M[i]), float(N_res[i]), float(eps0[i]), int(iters[i])) for i in range(theta.size)],
    )

    # --- Column: N-u (axial cyclic) using stateful fiber section ---
    col_state = _to_stateful(sec_col)
    L0 = 1.0  # [m] reference length for displacement u = eps0 * L0
    eps_hist = _history_from_cycles([2e-4, 4e-4, 6e-4], steps_per_half=steps_per_half)
    u = eps_hist * L0

    N = np.zeros_like(eps_hist)
    col_state.reset_state()
    for i in range(eps_hist.size):
        Ni, _Mi, *_ = col_state.trial_update(float(eps_hist[i]), 0.0)
        col_state.commit_trial()
        N[i] = float(Ni)

    fig, ax = plt.subplots(figsize=(7, 5))
    lc = add_time_gradient_line(ax, u, N, c=np.arange(u.size))
    add_colorbar(lc, ax, label="step")
    ax.set_xlabel("u [m] (axial)")
    ax.set_ylabel("N [N]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "problem5_fiber_col_Nu_gradient.png", dpi=180)
    plt.close(fig)

    _write_csv(
        out / "problem5_fiber_col_Nu.csv",
        header=["step", "eps0", "u_m", "N_N"],
        rows=[(i, float(eps_hist[i]), float(u[i]), float(N[i])) for i in range(eps_hist.size)],
    )

    # --- Column: N-M trajectory under combined history (eps0 + kappa) ---
    npts = int(3 * (4 * steps_per_half + 1))
    t = np.linspace(0.0, 6.0 * math.pi, npts)
    eps0_hist = 3.0e-4 * np.sin(t)
    kappa_hist = 0.02 * np.sin(2.0 * t + 0.3)  # [1/m], typical curvature level

    N2 = np.zeros_like(t)
    M2 = np.zeros_like(t)
    col_state.reset_state()
    for i in range(t.size):
        Ni, Mi, *_ = col_state.trial_update(float(eps0_hist[i]), float(kappa_hist[i]))
        col_state.commit_trial()
        N2[i] = float(Ni)
        M2[i] = float(Mi)

    fig, ax = plt.subplots(figsize=(7, 5))
    lc = add_time_gradient_line(ax, N2, M2, c=np.arange(t.size))
    add_colorbar(lc, ax, label="step")
    ax.set_xlabel("N [N]")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "problem5_fiber_col_NM_gradient.png", dpi=180)
    plt.close(fig)

    _write_csv(
        out / "problem5_fiber_col_NM.csv",
        header=["step", "t", "eps0", "kappa_1_per_m", "N_N", "M_Nm"],
        rows=[(i, float(t[i]), float(eps0_hist[i]), float(kappa_hist[i]), float(N2[i]), float(M2[i])) for i in range(t.size)],
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Problem 5: Fiber section interaction + cyclic fiber hinge tests")
    parser.add_argument("--mode", choices=["interaction", "tests", "all"], default="all")
    parser.add_argument("--steps-per-half", type=int, default=DEFAULT_STEPS_PER_HALF,
                        help="Resolution of cyclic histories. Total points per cycle ~ 4*steps_per_half+1.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    out = _outputs_dir()

    # --- Materials requested by the user ---
    fy = 420e6  # A420 (NCh) [Pa]
    fc = 20e6   # C20/25: use f_ck ≈ 20 MPa as reference [Pa]

    # --- Column section (example) ---
    b_col = cm_to_m(40.0)
    h_col = cm_to_m(40.0)
    cover_col = cm_to_m(4.0)

    phi = mm_to_m(16.0)
    n_bars_col = 4
    As_layer_col = float(n_bars_col) * _rebar_area(phi)
    rebar_layers_col = [
        (As_layer_col, cover_col, n_bars_col),
        (As_layer_col, h_col - cover_col, n_bars_col),
    ]
    As_tot_col = sum(As for As, _, _ in rebar_layers_col)

    sec_col = build_rc_rect_fiber_section_2d(
        b=b_col,
        h=h_col,
        cover=cover_col,
        fc=fc,
        fy=fy,
        eps_c0=0.002,
        eps_cu=0.0035,
        ny=70,
        nz=50,
        clustering="cosine",
        rebar_layers=rebar_layers_col,
    )

    # --- Beam section (example) ---
    b_beam = cm_to_m(30.0)
    h_beam = cm_to_m(50.0)
    cover_beam = cm_to_m(4.0)
    n_bars_beam = 3
    As_layer_beam = float(n_bars_beam) * _rebar_area(phi)
    rebar_layers_beam = [
        (As_layer_beam, cover_beam, n_bars_beam),
        (As_layer_beam, h_beam - cover_beam, n_bars_beam),
    ]
    As_tot_beam = sum(As for As, _, _ in rebar_layers_beam)

    sec_beam = build_rc_rect_fiber_section_2d(
        b=b_beam,
        h=h_beam,
        cover=cover_beam,
        fc=fc,
        fy=fy,
        eps_c0=0.002,
        eps_cu=0.0035,
        ny=70,
        nz=50,
        clustering="cosine",
        rebar_layers=rebar_layers_beam,
    )

    if args.mode in ("interaction", "all"):
        pts = sample_interaction_curve(sec_col, h=h_col, As_tot=As_tot_col, fy=fy, eps_cu=0.0035, n=200)
        pts = pts[np.isfinite(pts).all(axis=1)]
        hull = NMSurfacePolygon.from_points(pts)

        # Plot fiber mesh (quick sanity check)
        fig, ax = plt.subplots(figsize=(5, 5))
        ys = np.array([f.y for f in sec_col.fibers])
        zs = np.array([f.z for f in sec_col.fibers])
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
        (out / "problem5_fiber_interaction_summary.txt").write_text("\n".join(
                [
                    "Problem 5: Fiber section N-M interaction",
                    f"Column: b={b_col:.3f} m, h={h_col:.3f} m, cover={cover_col:.3f} m",
                    f"Beam:   b={b_beam:.3f} m, h={h_beam:.3f} m, cover={cover_beam:.3f} m",
                    f"fc={fc/1e6:.1f} MPa (C20/25 ref), fy={fy/1e6:.1f} MPa (A420)",
                    f"As_tot_col={As_tot_col*1e6:.1f} mm^2, As_tot_beam={As_tot_beam*1e6:.1f} mm^2",
                    f"n_points={pts.shape[0]}",
                ]
            ),
            encoding="utf-8",
        )

    if args.mode in ("tests", "all"):
        run_fiber_tests(
            sec_beam=sec_beam,
            sec_col=sec_col,
            h_beam=h_beam,
            h_col=h_col,
            out=out,
            steps_per_half=int(args.steps_per_half),
        )


if __name__ == "__main__":
    main()
