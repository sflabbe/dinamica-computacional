"""Plotting helpers for structure states."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection

from dc_solver.fem.model import Model
from dc_solver.fem.frame2d import rot2d


def beam_local_displacements(
    xi: np.ndarray,
    L: float,
    u_local: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return axial displacement, transverse displacement, and dv/dx along a beam in local coords."""
    s = np.asarray(xi, dtype=float)
    ux_i, uy_i, th_i, ux_j, uy_j, th_j = u_local
    u_ax = (1.0 - s) * ux_i + s * ux_j

    s2 = s * s
    s3 = s2 * s
    n1 = 1.0 - 3.0 * s2 + 2.0 * s3
    n2 = L * (s - 2.0 * s2 + s3)
    n3 = 3.0 * s2 - 2.0 * s3
    n4 = L * (-s2 + s3)
    v = n1 * uy_i + n2 * th_i + n3 * uy_j + n4 * th_j

    dn1 = -6.0 * s + 6.0 * s2
    dn2 = L * (1.0 - 4.0 * s + 3.0 * s2)
    dn3 = 6.0 * s - 6.0 * s2
    dn4 = L * (-2.0 * s + 3.0 * s2)
    dv_ds = dn1 * uy_i + dn2 * th_i + dn3 * uy_j + dn4 * th_j
    dv_dx = dv_ds / L
    return u_ax, v, dv_dx


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


def _normalize_field(field: Optional[str]) -> str:
    if field is None:
        return "both"
    field_key = str(field).strip().lower()
    aliases = {
        "u": "u",
        "disp": "u",
        "displacement": "u",
        "displacements": "u",
        "s": "s",
        "stress": "s",
        "stresses": "s",
        "both": "both",
        "combined": "both",
        "none": "none",
        "geometry": "none",
    }
    if field_key in {"u", "s", "both", "none"}:
        return field_key
    return aliases.get(field_key, "both")


def _field_cmap_label(field: str) -> Tuple[str, str]:
    if field == "u":
        return "viridis", "U, Magnitude [m]"
    if field == "s":
        return "plasma", "S, σ_abs,max [Pa]"
    return "viridis", ""


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


def _member_displacement_magnitudes(
    model: Model,
    u: Optional[np.ndarray],
    xi: np.ndarray,
) -> np.ndarray:
    if u is None:
        return np.array([], dtype=float)
    xy_ref = _node_xy(model, None)
    values = []
    for eb in model.beams:
        i, _ = eb.ni, eb.nj
        L, c, s = eb._geom()
        x_local = xi * L
        y_local = np.zeros_like(x_local)
        base = xy_ref[i]
        r = np.array([[c, -s], [s, c]])
        ref_pts = base + (r @ np.vstack([x_local, y_local])).T

        dofs = eb.dofs()
        u_g = u[dofs]
        u_l = rot2d(c, s) @ u_g
        u_ax, v, _ = beam_local_displacements(xi, L, u_l)
        def_pts = base + (r @ np.vstack([x_local + u_ax, v])).T
        disp = def_pts - ref_pts
        mags = np.linalg.norm(disp, axis=1)
        values.extend(0.5 * (mags[:-1] + mags[1:]))
    return np.asarray(values, dtype=float)


def _field_values_for_state(model: Model, u: Optional[np.ndarray], field: str) -> np.ndarray:
    if field == "u":
        xi = np.linspace(0.0, 1.0, 21)
        umag = nodal_displacement_magnitude(model, u)
        member_vals = _member_displacement_magnitudes(model, u, xi)
        if member_vals.size or umag.size:
            return np.concatenate([umag, member_vals]) if umag.size else member_vals
        return np.array([0.0], dtype=float)
    if field == "s":
        summary = member_stress_summary(model, u)
        values = np.array([row["sigma_abs_max"] for row in summary], dtype=float)
        return values if values.size else np.array([0.0], dtype=float)
    return np.array([], dtype=float)


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
    field: str = "both",
    norm: Optional[colors.Normalize] = None,
    add_colorbar: bool = True,
) -> None:
    field_kind = _normalize_field(field)
    xy_ref = _node_xy(model, None)
    u_plot = _scaled_displacements(model, u, scale) if u is not None else None
    xy = _node_xy(model, u_plot)

    xi = np.linspace(0.0, 1.0, 21)
    stress_summary = []
    if field_kind == "s":
        stress_summary = member_stress_summary(model, u)
    elif field_kind == "both" and u is not None:
        stress_summary = member_stress_summary(model, u)
    segments = []
    field_values = []
    for idx, eb in enumerate(model.beams):
        i, _ = eb.ni, eb.nj
        L, c, s = eb._geom()
        x_local = xi * L
        y_local = np.zeros_like(x_local)
        base = xy_ref[i]
        r = np.array([[c, -s], [s, c]])
        ref_pts = base + (r @ np.vstack([x_local, y_local])).T
        ax.plot(
            ref_pts[:, 0],
            ref_pts[:, 1],
            linewidth=1.0,
            color="0.85",
            zorder=1,
        )

        if u_plot is not None:
            dofs = eb.dofs()
            u_g = u_plot[dofs]
            u_l = rot2d(c, s) @ u_g
            u_ax, v, _ = beam_local_displacements(xi, L, u_l)
            def_pts = base + (r @ np.vstack([x_local + u_ax, v])).T
        else:
            def_pts = ref_pts

        if field_kind == "none":
            if u_plot is not None:
                ax.plot(def_pts[:, 0], def_pts[:, 1], linewidth=1.6, color="0.2", zorder=2)
            continue

        if field_kind in {"u", "s", "both"}:
            if field_kind == "both" and u_plot is None:
                continue
            if field_kind == "u" and u is not None:
                dofs_mag = eb.dofs()
                u_g_mag = u[dofs_mag]
                u_l_mag = rot2d(c, s) @ u_g_mag
                u_ax_mag, v_mag, _ = beam_local_displacements(xi, L, u_l_mag)
                def_pts_mag = base + (r @ np.vstack([x_local + u_ax_mag, v_mag])).T
                disp_mag = np.linalg.norm(def_pts_mag - ref_pts, axis=1)
            else:
                disp_mag = np.zeros_like(xi)

            for seg_idx, (a, b) in enumerate(zip(def_pts[:-1], def_pts[1:])):
                segments.append([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
                if field_kind == "u":
                    field_values.append(float(0.5 * (disp_mag[seg_idx] + disp_mag[seg_idx + 1])))
                elif stress_summary:
                    field_values.append(float(stress_summary[idx]["sigma_abs_max"]))

    if segments and field_kind in {"u", "s", "both"}:
        cmap_name, label = _field_cmap_label(field_kind)
        lc = LineCollection(
            segments,
            array=np.array(field_values),
            cmap=cmap_name,
            norm=norm,
            linewidths=2.2,
            zorder=2,
        )
        ax.add_collection(lc)
        if add_colorbar:
            if field_kind == "both":
                ax.figure.colorbar(lc, ax=ax, label="σ_max [Pa]")
            else:
                ax.figure.colorbar(lc, ax=ax, label=label)

    for hidx, h in enumerate(model.hinges):
        i = h.ni
        ax.scatter([xy[i, 0]], [xy[i, 1]], s=60, marker="o")
        ax.text(xy[i, 0], xy[i, 1], f"H{hidx}", fontsize=8, ha="left", va="bottom")

    if field_kind == "u":
        umag = nodal_displacement_magnitude(model, u)
        ax.scatter(xy[:, 0], xy[:, 1], s=26, c=umag, cmap="viridis", norm=norm, zorder=3)
    elif field_kind == "both":
        if u is None:
            ax.scatter(xy[:, 0], xy[:, 1], s=26, color="0.2", zorder=3)
        else:
            umag = nodal_displacement_magnitude(model, u)
            scatter = ax.scatter(xy[:, 0], xy[:, 1], s=26, c=umag, cmap="viridis", zorder=3)
            if add_colorbar:
                ax.figure.colorbar(scatter, ax=ax, label="|u| [m]")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=26, color="0.2", zorder=3)
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
    benchmark_kind: Optional[str] = None,
    benchmark_report: Optional[Dict[str, float]] = None,
    field: str = "both",
    shared_colorbar: bool = False,
) -> None:
    field_kind = _normalize_field(field)
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

    static_label = f"State 2: Static (gravity eq.)\n(scale={scale:.1f}×)"
    if benchmark_kind == "cantilever" and benchmark_report:
        static_label = (
            "State 2: Static\n"
            f"tip uy={benchmark_report['uy_tip']:.4e} m, "
            f"theta={benchmark_report['theta_tip']:.4e} rad (scale={scale:.1f}×)"
        )
    elif benchmark_kind == "simply_supported" and benchmark_report:
        static_label = (
            "State 2: Static\n"
            f"midspan uy={benchmark_report['uy_mid']:.4e} m (scale={scale:.1f}×)"
        )

    states = [
        ("State 1: Model (undeformed)", None),
        (static_label, u_static),
    ]
    has_dynamic = isinstance(t, np.ndarray) and t.size > 1
    if has_dynamic and u_snap is not None:
        snap_label = (
            f"State 3: Snapshot (drift ≥ {100.0 * float(snapshot_limit):.2f}%)"
            if snap_reached else
            f"State 3: Snapshot = peak drift (<{100.0 * float(snapshot_limit):.2f}%)"
        )
        snap_label += f"\n t={t_snap:.3f}s, drift={100.0*drift_snap:.2f}% (scale={scale:.1f}×)"
        states.append((snap_label, u_snap))
    if has_dynamic and u_peak is not None:
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

    norm = None
    if shared_colorbar and field_kind in {"u", "s"}:
        values = []
        for _, u_state in states:
            values.extend(_field_values_for_state(model, u_state, field_kind))
        values = np.array(values, dtype=float)
        if values.size:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            if math.isclose(vmin, vmax):
                vmax = vmin + 1e-12
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for ax, (title, u_state) in zip(axs, states):
        plot_structure_state(
            ax,
            model,
            u_state,
            title,
            scale=scale,
            field=field_kind,
            norm=norm,
            add_colorbar=not (shared_colorbar and field_kind in {"u", "s"}),
        )

    for ax in axs[len(states):]:
        ax.axis("off")

    if shared_colorbar and field_kind in {"u", "s"} and norm is not None:
        cmap_name, label = _field_cmap_label(field_kind)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        sm.set_array([])
        fig.colorbar(sm, ax=axs[:len(states)], label=label)

    fig.savefig(outfile, dpi=170)
    plt.close(fig)
