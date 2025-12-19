from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from dinamica_computacional.core.model import Model


def colored_line(ax, x, y, t, linewidth=2.0, cmap="viridis", label=None):
    x = np.asarray(x); y = np.asarray(y); t = np.asarray(t)
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, array=t[:-1], cmap=cmap, linewidth=linewidth)
    ax.add_collection(lc)
    if label:
        ax.plot([], [], label=label)
    ax.autoscale()
    return lc


def plot_deformed_mesh(ax, nodes, elements, u, dofmap, scale: float = 1.0, label_elems: bool = True):
    colors = {"COL": "tab:blue", "BEAM": "tab:orange"}
    for elem in elements:
        ni, nj = int(elem["ni"]), int(elem["nj"])
        xi, yi = nodes[ni]
        xj, yj = nodes[nj]
        ax.plot([xi, xj], [yi, yj], color="0.75", linewidth=1.0, zorder=1)

        if u is None:
            xi_d, yi_d = xi, yi
            xj_d, yj_d = xj, yj
        else:
            uxi, uyi = dofmap[ni]
            uxj, uyj = dofmap[nj]
            xi_d = xi + scale * float(u[uxi])
            yi_d = yi + scale * float(u[uyi])
            xj_d = xj + scale * float(u[uxj])
            yj_d = yj + scale * float(u[uyj])

        prop = str(elem.get("prop", ""))
        color = colors.get("COL" if "COL" in prop else "BEAM", "tab:blue")
        ax.plot([xi_d, xj_d], [yi_d, yj_d],
                color=color,
                linewidth=2.0, zorder=2)

        if label_elems:
            xm = 0.5 * (xi_d + xj_d)
            ym = 0.5 * (yi_d + yj_d)
            ax.text(xm, ym, f"E{elem['eid']}", fontsize=7, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_structure_state(ax, model: Model, u: Optional[np.ndarray], title: str, scale: float = 1.0):
    nodes = {i: (nd.x, nd.y) for i, nd in enumerate(model.nodes)}
    dofmap = {i: nd.dof_u for i, nd in enumerate(model.nodes)}
    plot_deformed_mesh(ax, nodes, model.elements_meta, u, dofmap, scale=scale, label_elems=True)

    for hidx, h in enumerate(model.hinges):
        i = h.nj
        xi, yi = nodes[i]
        if u is not None:
            ux, uy = dofmap[i]
            xi += scale * float(u[ux])
            yi += scale * float(u[uy])
        ax.scatter([xi], [yi], s=60, marker="o", zorder=3)
        ax.text(xi, yi, f"H{hidx}", fontsize=8, ha="left", va="bottom")

    ax.set_title(title)


def _snapshot_data(last: Dict[str, np.ndarray], snapshot_limit: float):
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


def plot_structure_states(model: Model, last: Dict[str, np.ndarray], drift_height: float, snapshot_limit: float = 0.04, outfile: str = "problem4_states_members.png"):
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

            k_snap, t_snap, drift_snap, snap_reached = _snapshot_data(last, snapshot_limit)
            if k_snap >= 0 and k_snap < u_hist.shape[0]:
                u_snap = u_hist[k_snap].copy()

    span = max(nd.x for nd in model.nodes) - min(nd.x for nd in model.nodes)
    span = float(span) if span > 0 else 1.0
    scale = 1.0
    u_for_scale = [u for u in (u_static, u_peak, u_snap) if u is not None]
    if u_for_scale:
        trans_dofs = [d for nd in model.nodes for d in nd.dof_u]
        umax = max(float(np.max(np.abs(u[trans_dofs]))) for u in u_for_scale if u.size)
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


def plot_results(model: Model, results, outfile_prefix: str = "problem4") -> None:
    if not results.dynamic_steps:
        return
    last_key = list(results.dynamic_steps.keys())[-1]
    last = results.dynamic_steps[last_key]

    drift_limit = 0.10
    snapshot_limit = float(last.get("snapshot_limit", 0.04))

    plot_structure_states(model, last, drift_height=max(nd.y for nd in model.nodes), snapshot_limit=snapshot_limit,
                          outfile=f"{outfile_prefix}_states_members.png")

    t = last.get("t", np.array([], dtype=float))
    drift = last.get("drift", np.array([], dtype=float))
    Vb = last.get("Vb", np.array([], dtype=float))
    ag = last.get("ag", np.array([], dtype=float))
    snap_idx, t_snap, _, _ = _snapshot_data(last, snapshot_limit)

    if t.size and ag.size:
        fig, ax = plt.subplots()
        ax.plot(t, ag)
        if snap_idx >= 0 and np.isfinite(t_snap):
            ax.axvline(t_snap, linestyle=":", linewidth=1.0, color="0.4")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("a_g [m/s²]")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{outfile_prefix}_last_ag.png", dpi=150)
        plt.close(fig)

    if t.size and drift.size:
        fig, ax = plt.subplots()
        ax.plot(t, 100.0 * drift)
        idxc = np.where(np.abs(drift) >= drift_limit)[0]
        if idxc.size:
            tc = float(t[int(idxc[0])])
            ax.axvline(tc, linestyle=":", linewidth=1.2)
            ax.text(tc, 0.0, f" t_collapse={tc:.3f}s", rotation=90, va="bottom", fontsize=8)
        ax.axhline(100.0 * drift_limit, linestyle="--")
        ax.axhline(-100.0 * drift_limit, linestyle="--")
        if snapshot_limit != drift_limit:
            ax.axhline(100.0 * snapshot_limit, linestyle=":", color="0.4", linewidth=1.2)
            ax.axhline(-100.0 * snapshot_limit, linestyle=":", color="0.4", linewidth=1.2)
        if snap_idx >= 0 and np.isfinite(t_snap):
            ax.axvline(t_snap, linestyle=":", linewidth=1.0, color="0.4")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("drift [%]")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(f"{outfile_prefix}_last_drift.png", dpi=150)
        plt.close(fig)

    if drift.size and Vb.size and t.size:
        fig, ax = plt.subplots()
        x = 100.0 * drift
        y = Vb / 1e3
        lc = colored_line(ax, x, y, t)
        if snap_idx >= 0 and snap_idx < x.size:
            ax.plot(x[snap_idx], y[snap_idx], marker="o", color="red", label="snapshot")
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("drift [%]")
        ax.set_ylabel("V_base [kN]")
        ax.grid(True)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("t [s]")
        fig.tight_layout()
        fig.savefig(f"{outfile_prefix}_last_Vb_drift.png", dpi=150)
        plt.close(fig)
