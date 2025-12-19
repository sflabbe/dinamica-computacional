"""Plotting helpers for structure states."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

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


def plot_structure_state(
    ax,
    model: Model,
    u: Optional[np.ndarray],
    title: str,
    scale: float = 1.0,
    show_node_ids: bool = True,
) -> None:
    if u is None:
        u_plot = None
    else:
        u_plot = u.copy()
        for nd in model.nodes:
            ux, uy = nd.dof_u
            u_plot[ux] *= scale
            u_plot[uy] *= scale

    xy = _node_xy(model, u_plot)

    for eidx, eb in enumerate(model.beams):
        i, j = eb.ni, eb.nj
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], linewidth=2.0)
        xm, ym = 0.5 * (xy[i, 0] + xy[j, 0]), 0.5 * (xy[i, 1] + xy[j, 1])
        ax.text(xm, ym, f"E{eidx}", fontsize=7, ha="center", va="center")

    for hidx, h in enumerate(model.hinges):
        i = h.ni
        ax.scatter([xy[i, 0]], [xy[i, 1]], s=60, marker="o")
        ax.text(xy[i, 0], xy[i, 1], f"H{hidx}", fontsize=8, ha="left", va="bottom")

    ax.scatter(xy[:, 0], xy[:, 1], s=12, marker=".", zorder=3)
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

    span = max(nd.x for nd in model.nodes) - min(nd.x for nd in model.nodes)
    span = float(span) if span > 0 else 1.0
    scale = 1.0
    u_for_scale = [u for u in (u_static, u_peak, u_snap) if u is not None]
    if u_for_scale:
        umax = max(float(np.max(np.abs(u))) for u in u_for_scale if u.size)
        if umax > 0:
            target = 0.10 * span
            scale = min(80.0, target / umax)

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
