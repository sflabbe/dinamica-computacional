"""CSV + plot exports for hinge time histories.

This module centralizes reusable post-processing used by the problem scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import math

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from dc_solver.fem.model import Model
from dc_solver.hinges.models import HingeNM2DElement, RotSpringElement, FiberRotSpringElement
from dc_solver.post.hysteresis_gradient import add_time_gradient_line, add_colorbar


def _axis_from_nodes(xi: float, yi: float, xj: float, yj: float) -> Tuple[float, float]:
    dx = float(xj - xi)
    dy = float(yj - yi)
    L = math.hypot(dx, dy)
    if L <= 1e-12:
        return 1.0, 0.0
    return dx / L, dy / L


def _get_attr_chain(obj: Any, names: List[str]) -> Any:
    cur = obj
    for n in names:
        cur = getattr(cur, n, None)
        if cur is None:
            return None
    return cur


def _try_get_surface_vertices(hinge_elem: Any) -> Optional[np.ndarray]:
    """Best-effort access to a polygonal N–M surface vertices array (n,2)."""
    candidates = [
        ["hinge", "hinge", "surface"],
        ["hinge", "surface"],
        ["surface"],
    ]
    for chain in candidates:
        surf = _get_attr_chain(hinge_elem, chain)
        if surf is None:
            continue
        verts = getattr(surf, "vertices", None)
        if verts is None:
            continue
        arr = np.asarray(verts, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 3:
            return arr
    return None


def plot_time_gradient(
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    hull_vertices: Optional[np.ndarray] = None,
    lw: float = 2.2,
) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if hull_vertices is not None and np.asarray(hull_vertices).size:
        hv = np.asarray(hull_vertices, dtype=float)
        hv = np.vstack([hv, hv[0]])
        ax.plot(hv[:, 0], hv[:, 1], "k-", lw=1.6, label="yield hull")
    t = np.asarray(t, dtype=float)
    if t.size:
        norm = Normalize(vmin=float(np.min(t)), vmax=float(np.max(t)))
        lc = add_time_gradient_line(ax, x, y, c=t, norm=norm, lw=lw)
        add_colorbar(lc, ax, label="t [s]")
    else:
        ax.plot(x, y, "-", lw=lw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if hull_vertices is not None and np.asarray(hull_vertices).size:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _save_csv(out_path: Path, columns: Dict[str, np.ndarray]) -> None:
    keys = list(columns.keys())
    arr = np.column_stack([np.asarray(columns[k], dtype=float) for k in keys])
    header = ",".join(keys)
    np.savetxt(Path(out_path), arr, delimiter=",", header=header, comments="")


def export_problem4_hinges(out: Path, model: Model, last: Dict[str, Any]) -> None:
    """Export per-hinge series + plots for Problem 4."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    hinge_hist = last.get("hinges", [])
    if not hinge_hist:
        return

    u_hist = np.asarray(last.get("u", []), dtype=float)
    t_full = np.asarray(last.get("t", []), dtype=float)
    if u_hist.ndim != 2 or t_full.ndim != 1:
        return

    n = min(len(hinge_hist), u_hist.shape[0] - 1, t_full.shape[0] - 1)
    if n <= 1:
        return
    time = t_full[1 : n + 1]

    hinges = getattr(model, "hinges", [])
    n_hinges = min(len(hinges), len(hinge_hist[0]))
    if n_hinges <= 0:
        return

    for ih in range(n_hinges):
        h = hinges[ih]
        infos = [hinge_hist[k][ih] if ih < len(hinge_hist[k]) else {} for k in range(n)]
        M = np.array([float(inf.get("M", float("nan"))) for inf in infos], dtype=float)

        if isinstance(h, HingeNM2DElement):
            dofs = np.asarray(h.dofs(), dtype=int)
            ni = model.nodes[h.ni]
            nj = model.nodes[h.nj]
            cx, cy = _axis_from_nodes(ni.x, ni.y, nj.x, nj.y)

            u_i = u_hist[1 : n + 1, dofs[0]]
            v_i = u_hist[1 : n + 1, dofs[1]]
            u_j = u_hist[1 : n + 1, dofs[3]]
            v_j = u_hist[1 : n + 1, dofs[4]]
            q0 = cx * (u_j - u_i) + cy * (v_j - v_i)
            theta = u_hist[1 : n + 1, dofs[5]] - u_hist[1 : n + 1, dofs[2]]

            N = np.array([float(inf.get("N", float("nan"))) for inf in infos], dtype=float)
            active_count = np.array([
                float(len(inf.get("active", []))) if hasattr(inf.get("active", []), "__len__") else float("nan")
                for inf in infos
            ], dtype=float)

            hull = _try_get_surface_vertices(h)

            plot_time_gradient(
                out / f"problem4_column_hinge_{ih}_NM_hull_gradient.png",
                N, M, time,
                xlabel="N [N]", ylabel="M [N-m]",
                title=f"Problem 4 — Column hinge #{ih}  N–M",
                hull_vertices=hull,
            )
            plot_time_gradient(
                out / f"problem4_column_hinge_{ih}_Fu_gradient.png",
                q0, N, time,
                xlabel="u_axial [m]", ylabel="N [N]",
                title=f"Problem 4 — Column hinge #{ih}  N–u",
            )
            plot_time_gradient(
                out / f"problem4_column_hinge_{ih}_Ft_gradient.png",
                time, N, time,
                xlabel="t [s]", ylabel="N [N]",
                title=f"Problem 4 — Column hinge #{ih}  N–t",
            )
            plot_time_gradient(
                out / f"problem4_column_hinge_{ih}_ut_gradient.png",
                time, q0, time,
                xlabel="t [s]", ylabel="u_axial [m]",
                title=f"Problem 4 — Column hinge #{ih}  u–t",
            )

            _save_csv(
                out / f"problem4_column_hinge_{ih}_series.csv",
                {
                    "t": time,
                    "u_axial": q0,
                    "theta": theta,
                    "N": N,
                    "M": M,
                    "active_count": active_count,
                },
            )
            continue

        if isinstance(h, (RotSpringElement, FiberRotSpringElement)):
            dofs = np.asarray(h.dofs(), dtype=int)
            theta = u_hist[1 : n + 1, dofs[5]] - u_hist[1 : n + 1, dofs[2]]
            ux_dof = model.nodes[h.ni].dof_u[0]
            u_joint = u_hist[1 : n + 1, ux_dof]

            plot_time_gradient(
                out / f"problem4_beam_hinge_{ih}_Mtheta_gradient.png",
                theta, M, time,
                xlabel="theta [rad]", ylabel="M [N-m]",
                title=f"Problem 4 — Beam hinge #{ih}  M–theta",
            )
            plot_time_gradient(
                out / f"problem4_beam_hinge_{ih}_Mt_gradient.png",
                time, M, time,
                xlabel="t [s]", ylabel="M [N-m]",
                title=f"Problem 4 — Beam hinge #{ih}  M–t",
            )
            plot_time_gradient(
                out / f"problem4_beam_hinge_{ih}_ut_gradient.png",
                time, u_joint, time,
                xlabel="t [s]", ylabel="u_x [m]",
                title=f"Problem 4 — Beam hinge #{ih}  u–t",
            )
            plot_time_gradient(
                out / f"problem4_beam_hinge_{ih}_thetatt_gradient.png",
                time, theta, time,
                xlabel="t [s]", ylabel="theta [rad]",
                title=f"Problem 4 — Beam hinge #{ih}  theta–t",
            )

            extras: Dict[str, np.ndarray] = {}
            for k in ["a", "N", "N_res", "eps0", "kappa", "iters", "dW_pl", "My"]:
                arr = np.array([float(inf.get(k, float("nan"))) for inf in infos], dtype=float)
                if np.isfinite(arr).any():
                    extras[k] = arr

            cols = {"t": time, "theta": theta, "M": M, "u_x": u_joint, **extras}
            _save_csv(out / f"problem4_beam_hinge_{ih}_series.csv", cols)


def export_nm_overlay_hull_gradient(
    out_path: Path,
    N: np.ndarray,
    M: np.ndarray,
    t: np.ndarray,
    hull_vertices: np.ndarray,
    *,
    title: str,
    xlabel: str = "N [N]",
    ylabel: str = "M [N-m]",
) -> None:
    """Convenience function for N–M paths with hull overlay."""
    plot_time_gradient(
        Path(out_path),
        np.asarray(N, dtype=float),
        np.asarray(M, dtype=float),
        np.asarray(t, dtype=float),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        hull_vertices=np.asarray(hull_vertices, dtype=float),
    )
