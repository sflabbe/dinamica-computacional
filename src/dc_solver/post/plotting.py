"""Plotting helpers for structure states."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dc_solver.fem.model import Model


def _node_xy(model: Model, u: Optional[np.ndarray] = None) -> np.ndarray:
    xy = np.array([[nd.x, nd.y] for nd in model.nodes], dtype=float)
    if u is None:
        return xy
    for i, nd in enumerate(model.nodes):
        ux, uy = nd.dof_u
        xy[i, 0] += float(u[ux])
        xy[i, 1] += float(u[uy])
    return xy


def nodal_displacement_magnitude(model: Model, u: Optional[np.ndarray]) -> np.ndarray:
    if u is None:
        return np.zeros(len(model.nodes))
    umag = np.zeros(len(model.nodes))
    for i, nd in enumerate(model.nodes):
        ux, uy = nd.dof_u
        umag[i] = math.hypot(float(u[ux]), float(u[uy]))
    return umag


def _scaled_displacements(model: Model, u: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    if u is None:
        return None
    u_plot = u.copy()
    for nd in model.nodes:
        ux, uy = nd.dof_u
        u_plot[ux] *= scale
        u_plot[uy] *= scale
    return u_plot


def _auto_scale(model: Model, states: Iterable[np.ndarray], max_scale: float = 200.0) -> float:
    xs = [nd.x for nd in model.nodes]
    ys = [nd.y for nd in model.nodes]
    size = max(max(xs) - min(xs), max(ys) - min(ys))
    size = float(size) if size > 0 else 1.0
    umax = 0.0
    for u in states:
        umag = nodal_displacement_magnitude(model, u)
        umax = max(umax, float(np.max(umag)) if umag.size else 0.0)
    if umax <= 0.0:
        return 1.0
    target = 0.10 * size
    return min(max_scale, target / umax)


def member_stress_summary(model: Model, u: Optional[np.ndarray]) -> list[dict]:
    if u is None:
        u = np.zeros(model.ndof())
    summary = []
    for eidx, eb in enumerate(model.beams):
        forces = eb.end_forces_local(u)
        n = forces["N"]
        mi = forces["Mi"]
        mj = forces["Mj"]
        A = eb.A
        I = eb.I
        h = math.sqrt(12.0 * I / A) if A > 0 and I > 0 else 0.0
        c = 0.5 * h
        sigma_n = n / A if A > 0 else 0.0
        sigma_b_i = mi * c / I if I > 0 else 0.0
        sigma_b_j = mj * c / I if I > 0 else 0.0
        sigma_max_i = sigma_n + abs(sigma_b_i)
        sigma_min_i = sigma_n - abs(sigma_b_i)
        sigma_max_j = sigma_n + abs(sigma_b_j)
        sigma_min_j = sigma_n - abs(sigma_b_j)
        sigma_abs_max = max(
            abs(sigma_max_i),
            abs(sigma_min_i),
            abs(sigma_max_j),
            abs(sigma_min_j),
        )
        summary.append({
            "element_index": eidx,
            "ni": eb.ni,
            "nj": eb.nj,
            "N": float(n),
            "Mi": float(mi),
            "Mj": float(mj),
            "sigma_abs_max": float(sigma_abs_max),
        })
    return summary


def write_member_stress_csv(model: Model, u: np.ndarray, path: str) -> None:
    rows = member_stress_summary(model, u)
    with open(path, "w", encoding="utf-8") as f:
        f.write("element_index,ni,nj,N,Mi,Mj,sigma_abs_max\n")
        for row in rows:
            f.write(
                f"{row['element_index']},{row['ni']},{row['nj']},"
                f"{row['N']},{row['Mi']},{row['Mj']},{row['sigma_abs_max']}\n"
            )


def plot_structure_state(
    ax,
    model: Model,
    u: Optional[np.ndarray],
    title: str,
    scale: float = 1.0,
    show_node_ids: bool = True,
) -> None:
    xy_ref = _node_xy(model, None)
    u_plot = _scaled_displacements(model, u, scale)
    xy = _node_xy(model, u_plot)

    for eb in model.beams:
        i, j = eb.ni, eb.nj
        ax.plot(
            [xy_ref[i, 0], xy_ref[j, 0]],
            [xy_ref[i, 1], xy_ref[j, 1]],
            linewidth=1.0,
            color="0.85",
            zorder=1,
        )

    stress_summary = member_stress_summary(model, u)
    segments = []
    stresses = []
    for row, eb in zip(stress_summary, model.beams):
        i, j = eb.ni, eb.nj
        segments.append([(xy[i, 0], xy[i, 1]), (xy[j, 0], xy[j, 1])])
        stresses.append(row["sigma_abs_max"])
    if segments:
        lc = LineCollection(segments, array=np.array(stresses), cmap="plasma", linewidths=2.2, zorder=2)
        ax.add_collection(lc)
        ax.figure.colorbar(lc, ax=ax, label="σ_max [Pa]")

    for hidx, h in enumerate(model.hinges):
        i = h.ni
        ax.scatter([xy[i, 0]], [xy[i, 1]], s=60, marker="o")
        ax.text(xy[i, 0], xy[i, 1], f"H{hidx}", fontsize=8, ha="left", va="bottom")

    umag = nodal_displacement_magnitude(model, u)
    scatter = ax.scatter(xy[:, 0], xy[:, 1], s=26, c=umag, cmap="viridis", zorder=3)
    ax.figure.colorbar(scatter, ax=ax, label="|u| [m]")
    if show_node_ids:
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
        offset = 0.01 * (span if span > 0 else 1.0)
        for i, (x, y) in enumerate(xy):
            ax.text(x + offset, y + offset, f"N{i}", fontsize=7, color="0.25")

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _snapshot_data(last: Dict[str, np.ndarray], snapshot_limit: float) -> Tuple[int, float, float, bool]:
    drift = last.get("drift", np.array([], dtype=float))
    t = last.get("t", np.array([], dtype=float))

    idx = int(last.get("snapshot_idx", -1))
    if idx < 0 or idx >= drift.size:
        idxs = np.where(np.abs(drift) >= snapshot_limit)[0] if drift.size else np.array([], dtype=int)
        if idxs.size:
            idx = int(idxs[0])
        elif drift.size:
            idx = int(np.argmax(np.abs(drift)))
        else:
            idx = -1

    drift_snap = float(drift[idx]) if (idx >= 0 and idx < drift.size) else float("nan")
    t_snap = float(t[idx]) if (idx >= 0 and idx < t.size) else float("nan")
    reached = bool(last.get("snapshot_reached", abs(drift_snap) >= snapshot_limit if np.isfinite(drift_snap) else False))
    return idx, t_snap, drift_snap, reached


def plot_structure_states(
    model: Model,
    last: Dict[str, np.ndarray],
    drift_height: float,
    snapshot_limit: Optional[float] = None,
    outfile: str = "problem4_states_members.png",
) -> None:
    if snapshot_limit is None:
        snapshot_limit = float(last.get("snapshot_limit", 0.04))
    u_hist = last.get("u", None)
    drift = last.get("drift", None)
    t = last.get("t", None)

    u_static = None
    u_peak = None
    t_peak = float("nan")
    drift_peak = float("nan")
    u_snap = None
    t_snap = float("nan")
    drift_snap = float("nan")
    snap_reached = False

    if isinstance(u_hist, np.ndarray) and u_hist.ndim == 2 and u_hist.shape[0] >= 1:
        u_static = u_hist[0].copy()
        if isinstance(drift, np.ndarray) and drift.size:
            k_peak = int(np.argmax(np.abs(drift)))
            u_peak = u_hist[k_peak].copy()
            drift_peak = float(drift[k_peak])
            if isinstance(t, np.ndarray) and t.size:
                t_peak = float(t[k_peak])

            if snapshot_limit is not None:
                k_snap, t_snap, drift_snap, snap_reached = _snapshot_data(last, snapshot_limit)
                if k_snap >= 0 and k_snap < u_hist.shape[0]:
                    u_snap = u_hist[k_snap].copy()

    u_for_scale = [u for u in (u_static, u_peak, u_snap) if u is not None]
    scale = _auto_scale(model, u_for_scale) if u_for_scale else 1.0

    states = [
        ("State 1: Model (undeformed)", None),
        (f"State 2: Static (gravity eq.)\n(scale={scale:.1f}×)", u_static),
    ]
    if u_snap is not None:
        snap_label = (
            f"State 3: Snapshot (drift ≥ {100.0 * float(snapshot_limit):.2f}%)"
            if snap_reached else
            f"State 3: Snapshot = peak drift (<{100.0 * float(snapshot_limit):.2f}%)"
        )
        snap_label += f"\n t={t_snap:.3f}s, drift={100.0*drift_snap:.2f}% (scale={scale:.1f}×)"
        states.append((snap_label, u_snap))
    states.append((
        f"State {len(states)+1}: Dynamic peak drift\n"
        f"t={t_peak:.3f}s, drift={100.0*drift_peak:.2f}% (scale={scale:.1f}×)",
        u_peak,
    ))

    n_states = len(states)
    if n_states <= 3:
        fig, axs = plt.subplots(1, n_states, figsize=(4.8 * n_states, 4.8), constrained_layout=True)
        axs = np.atleast_1d(axs)
    else:
        ncols = 2
        nrows = math.ceil(n_states / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), constrained_layout=True)
        axs = np.atleast_1d(axs).ravel()

    for ax, (title, u_state) in zip(axs, states):
        plot_structure_state(ax, model, u_state, title, scale=scale)

    for ax in axs[len(states):]:
        ax.axis("off")

    fig.savefig(outfile, dpi=170)
    plt.close(fig)
