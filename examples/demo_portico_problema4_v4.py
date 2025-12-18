"""examples.demo_portico_problema4

Fuente de la verdad (beam elements + rótulas concentradas) para el **Problema 4**.

Modelo (estable numéricamente):
  - Pórtico 1 piso / 1 vano: H=3 m, L=5 m.
  - **Beam2D** (Euler–Bernoulli 2D) elástico en columnas y viga.
  - **6 rótulas** concentradas (resortes rotacionales en serie con los beams):
      * Columnas: 2 por columna (base + cabeza) = 4 (N–M mediante My(N)).
      * Viga: 2 (extremos) = 2 (SHM M–θ con degradación simple).
  - Excitación: a_g(t)=A*cos(0.2*pi*t)*sin(4*pi*t), t<=10 s.
  - **IDA**: A = 0.1g, 0.2g, ... hasta colapso por drift.
  - **Colapso por drift**: 4%.
  - Postproceso: histeresis con gradiente temporal, N–M (±M), balance de energía y sensibilidad en dt.

Notas importantes:
  - Este modelo evita el problema típico de "zero-length con traslaciones libres".
    Las rótulas conectan **solo la rotación** (las traslaciones quedan compatibilizadas
    por construcción), lo que hace al sistema mucho más robusto.
  - Las columnas usan una aproximación estándar de interacción N–M:
      * Se construye una superficie (N,M) convexa desde la sección.
      * Para un axial N_ref (tomado al inicio del paso), se obtiene My(N_ref).
      * La rótula en columnas es M–θ perfecta-plástica con My dependiente de N.
    Esto mantiene el "M depende de N" sin introducir un DOF axial plástico adicional.

Requisitos:
  pip install numpy matplotlib

Ejecutar:
  python -m examples.demo_portico_problema4
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from plastic_hinge import RCSectionRect, RebarLayer, NMSurfacePolygon


# -----------------------------
# Plot helper: time gradient
# -----------------------------
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


# -----------------------------
# Plot: structure states (members + hinges)
# -----------------------------
def _node_xy(model: Model, u: Optional[np.ndarray] = None) -> np.ndarray:
    """Return nodal coordinates (nnode,2). If u given, apply translational DOFs."""
    xy = np.array([[nd.x, nd.y] for nd in model.nodes], dtype=float)
    if u is None:
        return xy
    for i, nd in enumerate(model.nodes):
        ux, uy = nd.dof_u
        xy[i, 0] += float(u[ux])
        xy[i, 1] += float(u[uy])
    return xy


def plot_structure_state(ax, model: Model, u: Optional[np.ndarray], title: str, scale: float = 1.0):
    """Plot member centerlines and hinge locations for a given displacement state."""
    # apply visual scale (only for translations)
    if u is None:
        u_plot = None
    else:
        u_plot = u.copy()
        for nd in model.nodes:
            ux, uy = nd.dof_u
            u_plot[ux] *= scale
            u_plot[uy] *= scale

    xy = _node_xy(model, u_plot)

    # beams (members)
    for eidx, eb in enumerate(model.beams):
        i, j = eb.ni, eb.nj
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], linewidth=2.5)
        xm, ym = 0.5 * (xy[i, 0] + xy[j, 0]), 0.5 * (xy[i, 1] + xy[j, 1])
        ax.text(xm, ym, f"B{eidx}", fontsize=8, ha="center", va="center")

    # hinges (locations)
    for hidx, h in enumerate(model.hinges):
        i = h.ni
        ax.scatter([xy[i, 0]], [xy[i, 1]], s=60, marker="o")
        ax.text(xy[i, 0], xy[i, 1], f"H{hidx}", fontsize=8, ha="left", va="bottom")

    # joints
    ax.scatter(xy[:, 0], xy[:, 1], s=12, marker=".", zorder=3)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_structure_states(model: Model, last: Dict[str, np.ndarray], drift_height: float, outfile: str = "problem4_states_members.png"):
    """State 1: undeformed model, State 2: gravity equilibrium, State 3: dynamic snapshot (peak drift)."""
    u_hist = last.get("u", None)
    drift = last.get("drift", None)
    t = last.get("t", None)

    u_static = None
    u_dyn = None
    t_dyn = float("nan")

    if isinstance(u_hist, np.ndarray) and u_hist.ndim == 2 and u_hist.shape[0] >= 1:
        u_static = u_hist[0].copy()
        if isinstance(drift, np.ndarray) and drift.size:
            k = int(np.argmax(np.abs(drift)))
            u_dyn = u_hist[k].copy()
            if isinstance(t, np.ndarray) and t.size:
                t_dyn = float(t[k])

    # auto scale for visibility
    span = max(nd.x for nd in model.nodes) - min(nd.x for nd in model.nodes)
    span = float(span) if span > 0 else 1.0
    scale = 1.0
    if u_dyn is not None:
        umax = float(np.max(np.abs(u_dyn))) if u_dyn.size else 0.0
        if umax > 0:
            target = 0.10 * span
            scale = min(80.0, target / umax)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    plot_structure_state(axs[0], model, None, "State 1: Model (undeformed)", scale=1.0)
    plot_structure_state(axs[1], model, u_static, f"State 2: Static (gravity eq.)\n(scale={scale:.1f}×)", scale=scale)
    title3 = f"State 3: Dynamic snapshot (peak drift)\n t={t_dyn:.3f}s (scale={scale:.1f}×)"
    plot_structure_state(axs[2], model, u_dyn, title3, scale=scale)

    fig.savefig(outfile, dpi=170)
    plt.close(fig)


# -----------------------------
# Diagnostics helpers
# -----------------------------
def format_snapshot(last: Dict[str, np.ndarray], drift_limit: float, extra: Optional[Dict[str, float]] = None) -> str:
    """Small multiline text block to embed in plots."""
    g = 9.81
    t = last.get("t", np.array([], dtype=float))
    drift = last.get("drift", np.array([], dtype=float))
    Vb = last.get("Vb", np.array([], dtype=float))

    pk_drift = float(np.max(np.abs(drift))) if drift.size else float("nan")
    pk_vb = float(np.max(np.abs(Vb))) if Vb.size else float("nan")

    idxc = np.where(np.abs(drift) >= drift_limit)[0] if drift.size else np.array([], dtype=int)
    tc = float(t[int(idxc[0])]) if (isinstance(t, np.ndarray) and t.size and idxc.size) else float("nan")

    it = last.get("iters", np.array([], dtype=int))
    it_max = int(np.max(it)) if isinstance(it, np.ndarray) and it.size else 0
    it_mean = float(np.mean(it)) if isinstance(it, np.ndarray) and it.size else 0.0

    A_g = float(last.get("A_g", float("nan")))
    dt = float(last.get("dt", float("nan")))
    alpha = float(last.get("alpha", float("nan")))
    rho_inf = float(last.get("rho_inf", float("nan")))
    gamma = float(last.get("gamma", float("nan")))
    beta = float(last.get("beta", float("nan")))
    zeta = float(last.get("zeta", float("nan")))
    T0 = float(last.get("T0", float("nan")))

    lines = [
        f"A = {A_g/g:.2f} g",
        f"dt = {1e3*dt:.3f} ms",
        f"T0 ≈ {T0:.3f} s, ζ = {zeta:.3f}",
        f"HHT-α: α={alpha:+.3f}, ρ∞={rho_inf:.3f}",
        f"γ={gamma:.3f}, β={beta:.3f}",
        f"Peak drift = {100*pk_drift:.2f}%  (limit {100*drift_limit:.2f}%)",
        f"Peak Vb = {pk_vb/1e3:.1f} kN",
        f"Newton iters: max={it_max}, mean={it_mean:.1f}",
    ]
    if idxc.size:
        lines.append(f"Collapse @ t={tc:.3f}s")
    if extra:
        for k, v in extra.items():
            try:
                lines.append(f"{k}: {float(v):.3g}")
            except Exception:
                lines.append(f"{k}: {v}")
    return "\n".join(lines)

def add_snapshot(ax, snapshot: str):
    ax.text(
        0.98, 0.98, snapshot,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def compute_energy_balance(last: Dict[str, np.ndarray], model: Model) -> Dict[str, np.ndarray]:
    """Energy balance using Euler–Lagrange bookkeeping on the *relative* DOFs.

    Returns cumulative work/dissipation and residual (numerical dissipation + error).
    """
    t = last["t"]
    u = last["u"]
    v = last.get("v", None)
    if v is None:
        # fallback: finite-diff velocities (rough)
        v = np.zeros_like(u)
        dt = float(t[1] - t[0]) if t.size > 1 else 1.0
        v[1:] = np.diff(u, axis=0) / dt

    M = model.mass_diag
    C = model.C_diag
    r = np.zeros_like(M)
    r[np.where(M > 0.0)[0]] = 1.0

    # Energies
    T = np.zeros(t.size)          # kinetic
    U_beam = np.zeros(t.size)     # elastic strain energy in beams
    U_hinge = np.zeros(t.size)    # elastic strain energy in hinges
    W_ext = np.zeros(t.size)      # external work (incl. gravity + quake)
    D_damp = np.zeros(t.size)     # viscous damping dissipation
    W_pl = np.zeros(t.size)       # plastic work in hinges (≥0)

    # precompute p(t)
    p = np.zeros((t.size, M.size))
    for k in range(t.size):
        p[k] = model.load_const - M * r * last["ag"][k]

    # power terms
    P_ext = np.einsum("ij,ij->i", p, v)
    P_damp = np.einsum("ij,j,ij->i", v, C, v)  # v^T C v (C diag)

    # hinge internal variables (trial/committed) come from last['hinges']
    hinge_steps = last.get("hinges", [])
    hinges = model.hinges

    for k in range(t.size):
        # kinetic
        T[k] = 0.5 * float(np.sum(M * v[k] * v[k]))

        # beam elastic energy
        Ub = 0.0
        for e in model.beams:
            dofs, k_g, _, _ = e.stiffness_and_force_global(u[k])
            ue = u[k, dofs]
            Ub += 0.5 * float(ue @ (k_g @ ue))
        U_beam[k] = Ub

        # hinge elastic energy (secant)
        Uh = 0.0
        for hi, h in enumerate(hinges):
            th_i = model.nodes[h.ni].dof_th
            th_j = model.nodes[h.nj].dof_th
            theta = float(u[k, th_j] - u[k, th_i])

            if k == 0:
                a_k = 0.0
                th_p_k = 0.0
            else:
                step = hinge_steps[k-1]
                if hi < len(step):
                    a_k = float(step[hi].get("a", 0.0))
                    th_p_k = float(step[hi].get("th_p", 0.0))
                else:
                    a_k = 0.0
                    th_p_k = 0.0

            th_el = theta - th_p_k
            if h.kind == "col_nm":
                K0 = float(h.col_hinge.k0)
            else:
                # degraded stiffness with current a_k
                K0 = float(h.beam_hinge.K0_0 * math.exp(-h.beam_hinge.cK * a_k))
            Uh += 0.5 * K0 * (th_el ** 2)

        U_hinge[k] = Uh

        # cumulative work/dissipation via trapezoid
        if k > 0:
            dt = float(t[k] - t[k-1])
            W_ext[k] = W_ext[k-1] + 0.5 * (P_ext[k] + P_ext[k-1]) * dt
            D_damp[k] = D_damp[k-1] + 0.5 * (P_damp[k] + P_damp[k-1]) * dt
            # plastic work from hinge return mapping bookkeeping
            dW = 0.0
            step = hinge_steps[k-1] if (k-1) < len(hinge_steps) else []
            for inf in step:
                dW += float(inf.get("dW_pl", 0.0))
            W_pl[k] = W_pl[k-1] + dW

    U = U_beam + U_hinge
    E_mech = T + U
    resid = W_ext - ((E_mech - E_mech[0]) + D_damp + W_pl)
    # normalise residual by total input work scale
    scale = max(1e-9, float(np.max(np.abs(W_ext))))
    resid_pct = 100.0 * resid / scale

    return {
        "t": t,
        "T": T,
        "U_beam": U_beam,
        "U_hinge": U_hinge,
        "U": U,
        "E_mech": E_mech,
        "W_ext": W_ext,
        "D_damp": D_damp,
        "W_pl": W_pl,
        "resid": resid,
        "resid_pct": resid_pct,
    }


def plot_energy_balance(last: Dict[str, np.ndarray], model: Model, drift_limit: float):
    eb = compute_energy_balance(last, model)

    snap = format_snapshot(
        last,
        drift_limit,
        extra={
            "Final |resid|/|Wext|max [%]": float(np.abs(eb["resid_pct"][-1])) if eb["resid_pct"].size else float("nan")
        },
    )

    # components
    fig, ax = plt.subplots()
    ax.plot(eb["t"], eb["T"], label="T (kinetic)")
    ax.plot(eb["t"], eb["U"], label="U (strain)")
    ax.plot(eb["t"], eb["E_mech"], label="T+U")
    ax.plot(eb["t"], eb["W_ext"], label="W_ext (input)")
    ax.plot(eb["t"], eb["D_damp"] + eb["W_pl"], label="D = damp + plastic")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Energy / Work [J]")
    ax.set_title("Energy balance (Euler–Lagrange bookkeeping)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    add_snapshot(ax, snap)
    fig.tight_layout()
    fig.savefig("problem4_last_energy_balance.png", dpi=150)
    plt.close(fig)

    # residual / numerical dissipation
    fig, ax = plt.subplots()
    ax.plot(eb["t"], eb["resid"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Residual [J]")
    ax.set_title("Energy residual = Wext - (ΔE + D)  (numerical dissipation + error)")
    ax.grid(True)
    add_snapshot(ax, snap)
    fig.tight_layout()
    fig.savefig("problem4_last_energy_residual.png", dpi=150)
    plt.close(fig)

    # percent
    fig, ax = plt.subplots()
    ax.plot(eb["t"], eb["resid_pct"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Residual [% of max|Wext|]")
    ax.set_title("Energy residual (percent)")
    ax.grid(True)
    add_snapshot(ax, snap)
    fig.tight_layout()
    fig.savefig("problem4_last_energy_residual_pct.png", dpi=150)
    plt.close(fig)

    return eb


def dt_sensitivity_analysis(H=3.0, L=5.0, T0=0.5, zeta=0.02,
                            A_g: float = 0.20,
                            drift_limit: float = 0.04,
                            alpha: float = -0.05,
                            t_end: float = 3.0,
                            dts: Tuple[float, ...] = (0.004, 0.002, 0.001, 0.0005)):
    """Run the same excitation with multiple dt values and plot key metrics."""
    g = 9.81
    A = float(A_g) * g

    rows = []
    ref = None
    for dt in dts:
        model, meta = build_portal_beam_hinge(H=H, L=L, T0=T0, zeta=zeta, P_gravity_total=1500e3)
        t = make_time(t_end, dt)
        ag = ag_fun(t, A)
        out = hht_alpha_newton(model, t, ag, drift_height=H, drift_limit=drift_limit, alpha=alpha, max_iter=60, tol=1e-6, verbose=False)
        out["A_g"] = float(A_g)
        out["zeta"] = float(zeta)
        out["T0"] = float(T0)
        eb = compute_energy_balance(out, model)

        pk_drift = float(np.max(np.abs(out["drift"])))
        pk_vb = float(np.max(np.abs(out["Vb"])))
        resid_end = float(eb["resid"][-1])
        resid_pct_end = float(eb["resid_pct"][-1])
        rows.append({
            "dt": dt,
            "peak_drift": pk_drift,
            "peak_Vb": pk_vb,
            "resid_end_J": resid_end,
            "resid_end_pct": resid_pct_end,
            "iters_mean": float(np.mean(out["iters"])) if out["iters"].size else 0.0,
            "iters_max": int(np.max(out["iters"])) if out["iters"].size else 0,
        })
        ref = ref or rows[-1]

    # save csv
    import csv
    with open("problem4_dt_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # plot metrics vs dt
    dt_arr = np.array([r["dt"] for r in rows])
    pkd = 100*np.array([r["peak_drift"] for r in rows])
    pkV = np.array([r["peak_Vb"] for r in rows]) / 1e3
    resP = np.array([r["resid_end_pct"] for r in rows])

    fig, ax = plt.subplots()
    ax.plot(1e3*dt_arr, pkd, marker="o", label="Peak drift [%]")
    ax.set_xlabel("dt [ms]")
    ax.set_ylabel("Peak drift [%]")
    ax.set_title(f"dt sensitivity (A={A_g:.2f}g, t_end={t_end:.1f}s)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("problem4_dt_sensitivity_peakdrift.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(1e3*dt_arr, pkV, marker="o", label="Peak Vbase [kN]")
    ax.set_xlabel("dt [ms]")
    ax.set_ylabel("Peak Vbase [kN]")
    ax.set_title(f"dt sensitivity (A={A_g:.2f}g, t_end={t_end:.1f}s)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("problem4_dt_sensitivity_peakVbase.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(1e3*dt_arr, resP, marker="o", label="Final residual [%]")
    ax.set_xlabel("dt [ms]")
    ax.set_ylabel("Final energy residual [% of max|Wext|]")
    ax.set_title(f"dt sensitivity (A={A_g:.2f}g, t_end={t_end:.1f}s)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("problem4_dt_sensitivity_energyresid.png", dpi=150)
    plt.close(fig)

    return rows

# -----------------------------
# Yield surface helpers
# -----------------------------
def mirror_section_about_middepth(sec: RCSectionRect) -> RCSectionRect:
    h = sec.h
    layers = [RebarLayer(As=l.As, y=h - l.y) for l in sec.layers]
    return RCSectionRect(
        b=sec.b, h=sec.h, fc=sec.fc, fy=sec.fy, Es=sec.Es,
        eps_c0=sec.eps_c0, eps_cu=sec.eps_cu,
        layers=layers, n_fibers=sec.n_fibers,
    )


def build_nm_surface(sec: RCSectionRect, npts: int = 90, tension_positive: bool = True) -> NMSurfacePolygon:
    # sample both bending senses and mirror to include +M and -M
    pts1 = sec.sample_interaction_curve(n=npts)
    pts2 = mirror_section_about_middepth(sec).sample_interaction_curve(n=npts)
    pts = np.vstack([pts1, pts2, pts1 * np.array([1.0, -1.0]), pts2 * np.array([1.0, -1.0])])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if tension_positive:
        pts = pts.copy()
        pts[:, 0] *= -1.0  # now tension +
    return NMSurfacePolygon.from_points(pts)


def moment_capacity_from_polygon(surface: NMSurfacePolygon, N: float) -> float:
    """Return max |M| for a given axial N by intersecting the convex polygon with a vertical line N=const.

    If N is outside the polygon N-range, it is clamped to the nearest bound.
    """
    V = np.asarray(surface.vertices, float)
    Nmin, Nmax = float(np.min(V[:, 0])), float(np.max(V[:, 0]))
    Nc = float(np.clip(N, Nmin, Nmax))

    Ms = []
    for i in range(V.shape[0]):
        a = V[i]
        b = V[(i + 1) % V.shape[0]]
        Na, Ma = float(a[0]), float(a[1])
        Nb, Mb = float(b[0]), float(b[1])
        # check if segment crosses N=Nc
        if (Na - Nc) == 0.0 and (Nb - Nc) == 0.0:
            # segment vertical at Nc: take both endpoints
            Ms.extend([Ma, Mb])
            continue
        if (Na - Nc) * (Nb - Nc) > 0:
            continue
        # linear interpolation for intersection
        if abs(Nb - Na) < 1e-18:
            continue
        t = (Nc - Na) / (Nb - Na)
        if -1e-12 <= t <= 1.0 + 1e-12:
            Mi = Ma + t * (Mb - Ma)
            Ms.append(float(Mi))

    if len(Ms) == 0:
        # fallback: take nearest vertex at clamped Nc
        j = int(np.argmin(np.abs(V[:, 0] - Nc)))
        return float(abs(V[j, 1]))
    return float(max(abs(min(Ms)), abs(max(Ms))))


# -----------------------------
# FE core
# -----------------------------
@dataclass
class Node:
    x: float
    y: float
    dof_u: Tuple[int, int]   # (ux, uy)
    dof_th: int              # theta


class DofManager:
    def __init__(self):
        self._next = 0

    def new_trans(self) -> Tuple[int, int]:
        ux = self._next; uy = self._next + 1
        self._next += 2
        return ux, uy

    def new_rot(self) -> int:
        th = self._next
        self._next += 1
        return th

    @property
    def ndof(self) -> int:
        return self._next


def rot2d(c, s):
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


@dataclass
class FrameElementLinear2D:
    ni: int
    nj: int
    E: float
    A: float
    I: float
    nodes: List[Node]

    def _geom(self):
        xi, yi = self.nodes[self.ni].x, self.nodes[self.ni].y
        xj, yj = self.nodes[self.nj].x, self.nodes[self.nj].y
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        c, s = dx / L, dy / L
        return L, c, s

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def k_local(self):
        L, c, s = self._geom()
        E, A, I = self.E, self.A, self.I
        k = np.zeros((6, 6))
        k_ax = E * A / L
        k[0, 0] = k_ax; k[0, 3] = -k_ax
        k[3, 0] = -k_ax; k[3, 3] = k_ax

        k11 = 12 * E * I / (L ** 3)
        k12 = 6 * E * I / (L ** 2)
        k22 = 4 * E * I / L
        k22b = 2 * E * I / L

        k[1, 1] = k11;   k[1, 2] = k12;   k[1, 4] = -k11;  k[1, 5] = k12
        k[2, 1] = k12;   k[2, 2] = k22;   k[2, 4] = -k12;  k[2, 5] = k22b
        k[4, 1] = -k11;  k[4, 2] = -k12;  k[4, 4] = k11;   k[4, 5] = -k12
        k[5, 1] = k12;   k[5, 2] = k22b;  k[5, 4] = -k12;  k[5, 5] = k22
        return k

    def stiffness_and_force_global(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        L, c, s = self._geom()
        T = rot2d(c, s)
        dofs = self.dofs()
        u_g = u[dofs]
        u_l = T @ u_g
        k_l = self.k_local()
        f_l = k_l @ u_l
        k_g = T.T @ k_l @ T
        f_g = T.T @ f_l

        # axial force in tension-positive convention
        k_ax = self.E * self.A / L
        du_ax = u_l[3] - u_l[0]
        N_tension = k_ax * du_ax

        return dofs, k_g, f_g, {"N": float(N_tension)}


@dataclass
class ColumnHingeNMRot:
    """Moment-rotation hinge with My dependent on an axial reference N_ref.

    Return mapping is 1D on M–θ with perfect plasticity + tiny post-yield tangent.
    """

    surface: NMSurfacePolygon
    k0: float
    alpha_post: float = 1e-4

    # committed state
    th_p_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0
    My_comm: float = 1.0

    def set_yield_from_N(self, N_ref: float):
        # My depends on axial force (tension +)
        self.My_comm = max(1e-6, moment_capacity_from_polygon(self.surface, N_ref))

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float]:
        """Incremental 1D return mapping (M–θ) para columna con **plástico ideal**.

        - Criterio: |M| <= My(N_ref)
        - Sin endurecimiento (ideal), pero con **tangente regularizada** k_reg = alpha_post*K0
          solo para mejorar la robustez numérica del Newton (no cambia el nivel de fluencia).

        Returns:
            (M_new, k_tan, th_p_new, a_new, M_plot)
        """
        K0 = float(self.k0)
        k_reg = max(float(self.alpha_post) * K0, 1e-12)

        M_trial = float(self.M_comm) + K0 * float(dth)
        f = abs(M_trial) - float(self.My_comm)

        if f <= 0.0:
            return M_trial, K0, self.th_p_comm, self.a_comm, M_trial

        # Perfect plastic corrector
        dg = f / max(K0, 1e-18)
        sgn = 1.0 if M_trial >= 0.0 else -1.0
        th_p_new = float(self.th_p_comm) + dg * sgn
        a_new = float(self.a_comm) + dg
        M_new = sgn * float(self.My_comm)

        # Regularized tangent (very small) to avoid rank-deficient K_eff
        k_tan = k_reg
        return M_new, k_tan, th_p_new, a_new, M_new





@dataclass
class SHMBeamHinge1D:
    K0_0: float
    My_0: float
    alpha_post: float = 0.02
    cK: float = 2.0
    cMy: float = 1.0

    th_p_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float]:
        K0 = self.K0_0 * math.exp(-self.cK * self.a_comm)
        My = self.My_0 * math.exp(-self.cMy * self.a_comm)
        Kp = self.alpha_post * K0
        H = (K0 * Kp) / max(K0 - Kp, 1e-18)

        M_trial = self.M_comm + K0 * dth
        f = abs(M_trial) - (My + H * self.a_comm)
        if f <= 0.0:
            return M_trial, K0, self.th_p_comm, self.a_comm, M_trial

        dg = f / (K0 + H)
        sgn = 1.0 if M_trial >= 0 else -1.0
        th_p_new = self.th_p_comm + dg * sgn
        a_new = self.a_comm + dg
        M_new = M_trial - K0 * dg * sgn
        k_tan = (K0 * H) / (K0 + H)
        return M_new, k_tan, th_p_new, a_new, M_new


@dataclass
class RotSpringElement:
    """Zero-length rotational spring between node i and j (only θ DOFs)."""
    ni: int
    nj: int
    kind: str  # "col_nm" or "beam_shm"
    col_hinge: Optional[ColumnHingeNMRot]
    beam_hinge: Optional[SHMBeamHinge1D]
    nodes: List[Node]

    # trial cache
    _trial: Dict = None

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        dofs = self.dofs()
        th_i = float(u_trial[dofs[2]])
        th_j = float(u_trial[dofs[5]])
        th_i_c = float(u_comm[dofs[2]])
        th_j_c = float(u_comm[dofs[5]])

        dth_inc = (th_j - th_i) - (th_j_c - th_i_c)

        # committed hinge state (for energy bookkeeping)
        if self.kind == "col_nm":
            assert self.col_hinge is not None
            a_old = float(self.col_hinge.a_comm)
            th_p_old = float(self.col_hinge.th_p_comm)
            M_old = float(self.col_hinge.M_comm)
            My_old = float(self.col_hinge.My_comm)

            M, kM, th_p_new, a_new, M_new = self.col_hinge.eval_increment(dth_inc)
            trial_state = {"th_p_new": th_p_new, "a_new": a_new, "M_new": M_new, "M": float(M)}

            dg = float(a_new) - a_old
            dth_p = float(th_p_new) - th_p_old
            dW_pl = abs(float(M_new)) * dg  # plastic work increment (≥0)
            K0 = float(self.col_hinge.k0)

            info_extra = {
                "My": My_old,
                "K0": K0,
                "dg": dg,
                "dtheta_p": dth_p,
                "M_old": M_old,
                "M_new": float(M_new),
                "a": float(a_new),
                "th_p": float(th_p_new),
                "dW_pl": float(dW_pl),
            }

        elif self.kind == "beam_shm":
            assert self.beam_hinge is not None
            a_old = float(self.beam_hinge.a_comm)
            th_p_old = float(self.beam_hinge.th_p_comm)
            M_old = float(self.beam_hinge.M_comm)

            # current degraded elastic stiffness at committed state
            K0 = float(self.beam_hinge.K0_0 * math.exp(-self.beam_hinge.cK * a_old))
            My = float(self.beam_hinge.My_0 * math.exp(-self.beam_hinge.cMy * a_old))

            M, kM, th_p_new, a_new, M_new = self.beam_hinge.eval_increment(dth_inc)
            trial_state = {"th_p_new": th_p_new, "a_new": a_new, "M_new": M_new, "M": float(M)}

            dg = float(a_new) - a_old
            dth_p = float(th_p_new) - th_p_old
            dW_pl = abs(float(M_new)) * dg

            info_extra = {
                "My": My,
                "K0": K0,
                "dg": dg,
                "dtheta_p": dth_p,
                "M_old": M_old,
                "M_new": float(M_new),
                "a": float(a_new),
                "th_p": float(th_p_new),
                "dW_pl": float(dW_pl),
            }
        else:
            raise ValueError("Unknown hinge kind")

        # Bm maps [.., th_i, .., th_j] -> dθ
        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = kM * (Bm.T @ Bm)
        f_l = (Bm.T * M).reshape(6)

        self._trial = trial_state
        info = {"dtheta": float(dth_inc), "M": float(M)}
        info.update(info_extra)
        return k_l, f_l, info

    def commit(self):
        if self._trial is None:
            return
        if self.kind == "col_nm":
            self.col_hinge.th_p_comm = self._trial["th_p_new"]
            self.col_hinge.a_comm = self._trial["a_new"]
            self.col_hinge.M_comm = self._trial["M_new"]
        elif self.kind == "beam_shm":
            self.beam_hinge.th_p_comm = self._trial["th_p_new"]
            self.beam_hinge.a_comm = self._trial["a_new"]
            self.beam_hinge.M_comm = self._trial["M_new"]


@dataclass
class Model:
    nodes: List[Node]
    beams: List[FrameElementLinear2D]
    hinges: List[RotSpringElement]
    fixed_dofs: np.ndarray
    mass_diag: np.ndarray
    C_diag: np.ndarray
    load_const: np.ndarray

    # mapping: which beam gives N_ref for which column hinge
    col_hinge_groups: List[Tuple[int, int, int]]  # (hinge_idx, beam_idx, sign)

    def ndof(self) -> int:
        return int(self.mass_diag.size)

    def free_dofs(self) -> np.ndarray:
        all_dofs = np.arange(self.ndof(), dtype=int)
        mask = np.ones(self.ndof(), dtype=bool)
        mask[self.fixed_dofs] = False
        return all_dofs[mask]

    def reset_state(self):
        for h in self.hinges:
            h._trial = None
            if h.kind == "col_nm":
                h.col_hinge.th_p_comm = 0.0
                h.col_hinge.M_comm = 0.0
            else:
                h.beam_hinge.th_p_comm = 0.0
                h.beam_hinge.a_comm = 0.0
                h.beam_hinge.M_comm = 0.0

    def update_column_yields(self, u_comm: np.ndarray):
        """Update My(N) for each column hinge based on the *committed* axial forces."""
        # compute axial N per beam
        N_beam = []
        for b in self.beams:
            _, _, _, meta = b.stiffness_and_force_global(u_comm)
            N_beam.append(meta["N"])  # tension +
        for hinge_idx, beam_idx, sign in self.col_hinge_groups:
            h = self.hinges[hinge_idx]
            Nref = float(sign) * float(N_beam[beam_idx])
            h.col_hinge.set_yield_from_N(Nref)

    def assemble(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        nd = self.ndof()
        K = np.zeros((nd, nd))
        R = np.zeros(nd)
        info = {"hinges": []}

        for e in self.beams:
            dofs, k_g, f_g, _ = e.stiffness_and_force_global(u_trial)
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_g[a, b]

        for h in self.hinges:
            k_l, f_l, inf = h.eval_trial(u_trial, u_comm)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_l[a, b]
            info["hinges"].append(inf)

        fd = self.free_dofs()
        return K[np.ix_(fd, fd)], R[fd], info

    def commit(self):
        for h in self.hinges:
            h.commit()

    def base_shear(self, u: np.ndarray) -> float:
        # sum reactions at base ux DOFs (node0, node1)
        base_ux = [self.nodes[0].dof_u[0], self.nodes[1].dof_u[0]]
        # internal force vector at full u
        nd = self.ndof()
        R = np.zeros(nd)
        for e in self.beams:
            dofs, _, f_g, _ = e.stiffness_and_force_global(u)
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
        for h in self.hinges:
            _, f_l, _ = h.eval_trial(u, u)
            dofs = h.dofs()
            for a, ia in enumerate(dofs):
                R[ia] += f_l[a]
        Vb = 0.0
        for d in base_ux:
            Vb += -R[d]
        return float(Vb)


# -----------------------------
# Solver: HHT-alpha + Newton
# -----------------------------
def hht_alpha_newton(model: Model,
                     t: np.ndarray,
                     ag: np.ndarray,
                     drift_height: float,
                     drift_limit: float = 0.04,
                     alpha: float = -0.05,
                     max_iter: int = 40,
                     tol: float = 1e-6,
                     verbose: bool = False) -> Dict[str, np.ndarray]:
    if not (-1.0/3.0 - 1e-12 <= alpha <= 1e-12):
        raise ValueError("HHT-alpha requires alpha in [-1/3, 0].")

    model.reset_state()
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size

    # --- Static gravity equilibrium ---
    u = np.zeros(nd)
    u_free = u[fd].copy()
    for it in range(60):
        u_trial = u.copy(); u_trial[fd] = u_free
        model.update_column_yields(u_trial)  # update My under current N
        K, Rint, _ = model.assemble(u_trial, u)  # use u as "comm" (start)
        res = model.load_const[fd] - Rint
        if np.linalg.norm(res) < 1e-10 * max(1.0, np.linalg.norm(model.load_const[fd])):
            u = u_trial.copy()
            model.commit()
            break
        du = np.linalg.solve(K + 1e-14*np.eye(nf), res)
        u_free += du
    else:
        raise RuntimeError("No converge el paso estático de gravedad.")

    u_n = u.copy()
    v_n = np.zeros(nd)
    a_n = np.zeros(nd)

    M = model.mass_diag
    C = model.C_diag

    dt = float(t[1] - t[0])
    gamma = 0.5 - alpha
    beta = 0.25 * (1.0 - alpha)**2
    rho_inf = (1.0 + alpha) / (1.0 - alpha)
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)

    # influence vector: only DOFs with mass should respond to ground accel
    r = np.zeros(nd)
    r[np.where(M > 0.0)[0]] = 1.0

    u_hist = np.zeros((t.size, nd))
    v_hist = np.zeros((t.size, nd))
    a_hist = np.zeros((t.size, nd))
    drift = np.zeros(t.size)
    Vb = np.zeros(t.size)
    iters = np.zeros(t.size-1, dtype=int)

    hinge_hist: List[List[Dict]] = []

    # helper drift: top joints are node2 and node3 (physical)
    ux2 = model.nodes[2].dof_u[0]
    ux3 = model.nodes[3].dof_u[0]

    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift[0] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
    Vb[0] = model.base_shear(u_n)

    p_const = model.load_const.copy()

    for n in range(t.size - 1):
        # update My(N) with committed u_n
        model.update_column_yields(u_n)

        p_n = p_const - M * r * ag[n]
        p_np1 = p_const - M * r * ag[n+1]
        p_alpha = (1.0 + alpha) * p_np1 - alpha * p_n

        # internal at n
        _, Rint_n, _ = model.assemble(u_n, u_n)

        # predictors
        u_pred = u_n + dt * v_n + dt * dt * (0.5 - beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        # initial guess
        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy()

        # Newton iterations
        for it in range(max_iter):
            u_trial = u_comm_step.copy(); u_trial[fd] = u_free

            K_tan, Rint, inf = model.assemble(u_trial, u_comm_step)

            # HHT-alpha effective accel/vel
            a_trial = a0 * (u_trial - u_pred)
            v_trial = v_pred + (gamma * dt) * a_trial

            res = p_alpha[fd] - ( (1.0 + alpha) * Rint - alpha * Rint_n + C[fd] * v_trial[fd] + M[fd] * a_trial[fd] )

            # robust norm scaling
            scale = 1.0 + np.linalg.norm(p_alpha[fd])
            if np.linalg.norm(res) <= tol * scale:
                # commit
                u_n = u_trial
                v_n = v_trial
                a_n = a_trial
                model.commit()
                iters[n] = it + 1
                hinge_hist.append(inf["hinges"])
                break

            K_eff = ( (1.0 + alpha) * K_tan
                      + np.diag(M[fd] * a0)
                      + np.diag(C[fd] * a1) )
            try:
                du = np.linalg.solve(K_eff + 1e-14*np.eye(nf), res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K_eff + 1e-14*np.eye(nf), res, rcond=None)[0]
            u_free += du

        else:
            raise RuntimeError(f"No converge en paso {n+1} / t={t[n+1]:.3f}s (HHT-alpha).")

        u_hist[n+1] = u_n
        v_hist[n+1] = v_n
        a_hist[n+1] = a_n
        drift[n+1] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
        Vb[n+1] = model.base_shear(u_n)

        if abs(drift[n+1]) >= drift_limit:
            if verbose:
                print(f"COLLAPSE by drift >= {100*drift_limit:.1f}% at t={t[n+1]:.3f}s")
            # truncate
            u_hist = u_hist[:n+2]
            v_hist = v_hist[:n+2]
            a_hist = a_hist[:n+2]
            drift = drift[:n+2]
            Vb = Vb[:n+2]
            t = t[:n+2]
            ag = ag[:n+2]
            iters = iters[:n+1]
            break

    return {
        "t": t,
        "ag": ag,
        "u": u_hist,
        "v": v_hist,
        "a": a_hist,
        "drift": drift,
        "Vb": Vb,
        "iters": iters,
        "hinges": hinge_hist,
        "dt": dt,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "rho_inf": rho_inf,
    }


# -----------------------------
# Model builder
# -----------------------------
def build_portal_beam_hinge(H: float = 3.0,
                            L: float = 5.0,
                            T0: float = 0.5,
                            zeta: float = 0.02,
                            P_gravity_total: float = 1500e3) -> Tuple[Model, Dict]:
    dm = DofManager()

    # physical joint nodes (unique translations & rotations)
    n0 = Node(0.0, 0.0, dm.new_trans(), dm.new_rot())
    n1 = Node(L,   0.0, dm.new_trans(), dm.new_rot())
    n2 = Node(0.0, H,   dm.new_trans(), dm.new_rot())
    n3 = Node(L,   H,   dm.new_trans(), dm.new_rot())

    nodes: List[Node] = [n0, n1, n2, n3]

    # hinge-side nodes: share translations with the corresponding joint, but have own rotation
    def aux_at(j: int) -> int:
        nj = nodes[j]
        na = Node(nj.x, nj.y, nj.dof_u, dm.new_rot())
        nodes.append(na)
        return len(nodes) - 1

    # column hinge nodes
    i0L = aux_at(0)  # base left (member side)
    i2L = aux_at(2)  # top left  (member side)
    i1R = aux_at(1)
    i3R = aux_at(3)
    # beam hinge nodes
    i2B = aux_at(2)
    i3B = aux_at(3)

    # section placeholders (puedes reemplazar por las del Problema 2)
    fc = 30e6
    fy = 420e6
    Es = 200e9

    # Column section (rectangular RC)
    b_col, h_col = 0.30, 0.40
    layers_col = [
        RebarLayer(As=4 * (math.pi*(16e-3/2)**2), y=0.05),
        RebarLayer(As=4 * (math.pi*(16e-3/2)**2), y=h_col-0.05),
    ]
    # Mantener esto liviano para que el ejemplo corra rápido en Windows.
    sec_col = RCSectionRect(b=b_col, h=h_col, fc=fc, fy=fy, Es=Es, layers=layers_col, n_fibers=60)
    surf_col = build_nm_surface(sec_col, npts=60, tension_positive=True)

    # Beam section (for elastic EI)
    b_beam, h_beam = 0.30, 0.50
    A_beam = b_beam * h_beam
    I_beam = b_beam * (h_beam**3) / 12.0

    A_col = b_col * h_col
    I_col = b_col * (h_col**3) / 12.0

    # Elastic properties (tune if needed)
    E = 30e9

    # Beams (elastic): connect member-side nodes
    beams = [
        FrameElementLinear2D(i0L, i2L, E=E, A=A_col, I=I_col, nodes=nodes),  # left column
        FrameElementLinear2D(i1R, i3R, E=E, A=A_col, I=I_col, nodes=nodes),  # right column
        FrameElementLinear2D(i2B, i3B, E=E, A=A_beam, I=I_beam, nodes=nodes),  # top beam
    ]

    # Hinge initial rotational stiffness ~ (EI/L)*c
    k_col0 = 6.0 * E * I_col / H
    k_beam0 = 6.0 * E * I_beam / L

    # Hinges:
    hinges: List[RotSpringElement] = []
    # Columns: (joint rotation) -- spring -- (member rotation)
    # left base: node0 <-> i0L
    hinges.append(RotSpringElement(0, i0L, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    hinges.append(RotSpringElement(2, i2L, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    hinges.append(RotSpringElement(1, i1R, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    hinges.append(RotSpringElement(3, i3R, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    # Beam hinges: SHM moment hinges
    shm_left = SHMBeamHinge1D(K0_0=k_beam0, My_0=400e3)   # placeholder
    shm_right = SHMBeamHinge1D(K0_0=k_beam0, My_0=400e3)
    hinges.append(RotSpringElement(2, i2B, "beam_shm", None, shm_left, nodes))
    hinges.append(RotSpringElement(3, i3B, "beam_shm", None, shm_right, nodes))

    # fixed DOFs: base nodes 0 and 1 fully fixed
    fixed = np.array([
        nodes[0].dof_u[0], nodes[0].dof_u[1], nodes[0].dof_th,
        nodes[1].dof_u[0], nodes[1].dof_u[1], nodes[1].dof_th,
    ], dtype=int)

    nd = dm.ndof
    mass = np.zeros(nd)
    C = np.zeros(nd)
    p0 = np.zeros(nd)

    # gravity loads at top translations (uy)
    p0[nodes[2].dof_u[1]] += -0.5 * P_gravity_total
    p0[nodes[3].dof_u[1]] += -0.5 * P_gravity_total

    # temporary model (mass will be calibrated)
    model = Model(nodes=nodes, beams=beams, hinges=hinges,
                  fixed_dofs=fixed, mass_diag=mass, C_diag=C, load_const=p0,
                  col_hinge_groups=[
                      # map each column hinge to its column beam
                      (0, 0, +1), (1, 0, +1),
                      (2, 1, +1), (3, 1, +1),
                  ])

    # --- Calibrate mass to hit T0 ---
    K_story = story_stiffness_linear(model)
    omega0 = 2.0 * math.pi / T0
    M_total = K_story / (omega0**2)

    # assign mass to top ux DOFs only
    mass[nodes[2].dof_u[0]] = 0.5 * M_total
    mass[nodes[3].dof_u[0]] = 0.5 * M_total
    # mass-proportional damping on those DOFs
    C[:] = 2.0 * zeta * omega0 * mass

    meta = {"K_story": K_story, "M_total": M_total, "T0": T0, "omega0": omega0,
            "section_col": sec_col, "surface_col": surf_col}
    return model, meta


def story_stiffness_linear(model: Model) -> float:
    """Compute lateral story stiffness (elastic) via static solve with unit lateral load."""
    nd = model.ndof()
    fd = model.free_dofs()

    u_comm = np.zeros(nd)
    u_trial = np.zeros(nd)

    # set column yields (doesn't matter in elastic range, but keeps things initialized)
    model.update_column_yields(u_comm)

    K, _, _ = model.assemble(u_trial, u_comm)

    f = np.zeros(nd)
    ux2 = model.nodes[2].dof_u[0]
    ux3 = model.nodes[3].dof_u[0]
    f[ux2] = 0.5
    f[ux3] = 0.5
    f_free = f[fd]

    u_free = np.linalg.solve(K + 1e-14*np.eye(fd.size), f_free)
    u = np.zeros(nd); u[fd] = u_free
    u_top = 0.5 * (u[ux2] + u[ux3])
    return float(1.0 / u_top)


# -----------------------------
# IDA runner + plotting
# -----------------------------
def make_time(t_end: float, dt: float) -> np.ndarray:
    return np.arange(0.0, t_end + 1e-12, dt)


def ag_fun(t: np.ndarray, A: float) -> np.ndarray:
    return A * np.cos(0.2 * np.pi * t) * np.sin(4.0 * np.pi * t)


def run_incremental_amplitudes(H=3.0, L=5.0, T0=0.5, zeta=0.02,
                               drift_limit=0.04,
                               amps_g=np.arange(0.1, 2.1, 0.1),
                               t_end=10.0,
                               base_dt=0.002,
                               dt_min=0.00025,
                               alpha=-0.05):
    g = 9.81
    model, meta = build_portal_beam_hinge(H=H, L=L, T0=T0, zeta=zeta, P_gravity_total=1500e3)

    print(f"Modelo: hinges N-M (columnas) = 4, hinges M-θ (viga) = 2, total = 6")
    print(f"Calibracion elastica: K_story={meta['K_story']:.3e} N/m,  M_total={meta['M_total']:.3e} kg  (T0~{T0}s)")
    rho_inf = (1.0 + alpha) / (1.0 - alpha)
    print(f"HHT-α: alpha={alpha:+.3f} -> high-frequency spectral radius rho_inf={rho_inf:.3f} (numerical damping)")
    print(f"IDA: drift_limit={100*drift_limit:.2f}%, base_dt={base_dt:.6f}s, dt_min={dt_min:.6f}s\n")

    peak_drifts = []
    last = None

    for A_g in amps_g:
        A = float(A_g) * g
        print(f"--- Run A={A_g:.1f}g ---")

        dt = base_dt
        while True:
            t = make_time(t_end, dt)
            ag = ag_fun(t, A)
            try:
                out = hht_alpha_newton(model, t, ag, drift_height=H,
                                       drift_limit=drift_limit, alpha=alpha,
                                       max_iter=50, tol=1e-6, verbose=False)
                out["A_g"] = float(A_g)
                out["zeta"] = float(zeta)
                out["T0"] = float(T0)
                last = out
                break
            except RuntimeError as e:
                if dt <= dt_min + 1e-15:
                    print(f"Numeric failure at A={A_g:.1f}g (dt={dt:.6f}s): {e}")
                    return peak_drifts, amps_g[:len(peak_drifts)], last, model, meta
                dt *= 0.5
                print(f"  -> Retry due to non-convergence: dt -> {dt:.6f}s")

        pk = float(np.max(np.abs(out["drift"])))
        peak_drifts.append(pk)
        print(f"Peak drift = {100*pk:.2f}%")
        if out["iters"].size:
            print(f"Newton iters: max={int(np.max(out['iters']))}, mean={float(np.mean(out['iters'])):.1f} | alpha={out.get('alpha', float('nan')):+.3f}, gamma={out.get('gamma', float('nan')):.3f}, beta={out.get('beta', float('nan')):.3f}, rho_inf={out.get('rho_inf', float('nan')):.3f}, dt={out.get('dt', float('nan'))*1e3:.3f} ms\n")
        else:
            print()

        if pk >= drift_limit:
            print(f"COLLAPSE by drift >= {100*drift_limit:.2f}%")
            break

    return peak_drifts, amps_g[:len(peak_drifts)], last, model, meta


def plot_results(last: Dict[str, np.ndarray], model: Model, meta: Dict, drift_limit: float = 0.04):
    """Generate all standard plots for the last successful IDA run."""
    if last is None:
        return

    t = last.get("t", np.array([], dtype=float))
    drift = last.get("drift", np.array([], dtype=float))
    Vb = last.get("Vb", np.array([], dtype=float))
    ag = last.get("ag", np.array([], dtype=float))
    snap = format_snapshot(last, drift_limit)

    # --- State plots (members + hinges) ---
    try:
        H = float(meta.get("H", 1.0))
        plot_structure_states(model, last, drift_height=H, outfile="problem4_states_members.png")
    except Exception:
        pass

    # --- a_g(t) ---
    if t.size and ag.size:
        fig, ax = plt.subplots()
        ax.plot(t, ag)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("a_g [m/s²]")
        ax.grid(True)
        add_snapshot(ax, snap)
        fig.tight_layout()
        fig.savefig("problem4_last_ag.png", dpi=150)
        plt.close(fig)

    # --- drift(t) ---
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
        ax.set_xlabel("t [s]")
        ax.set_ylabel("drift [%]")
        ax.grid(True)
        add_snapshot(ax, snap)
        fig.tight_layout()
        fig.savefig("problem4_last_drift.png", dpi=150)
        plt.close(fig)

    # --- Vb–drift hysteresis with time gradient ---
    if drift.size and Vb.size and t.size:
        fig, ax = plt.subplots()
        x = 100.0 * drift
        y = Vb / 1e3  # kN
        lc = colored_line(x, y, t)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlabel("drift [%]")
        ax.set_ylabel("V_base [kN]")
        ax.grid(True)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("t [s]")
        add_snapshot(ax, snap)
        fig.tight_layout()
        fig.savefig("problem4_last_Vb_drift.png", dpi=150)
        plt.close(fig)

    # --- Beam hinge M–θ loops (time gradient) ---
    try:
        hinges_hist = last.get("hinges", [])
        if isinstance(hinges_hist, list) and len(hinges_hist) > 2:
            nh = len(hinges_hist[0])
            dth = np.zeros((len(hinges_hist), nh), dtype=float)
            M = np.zeros((len(hinges_hist), nh), dtype=float)
            for k, hinfs in enumerate(hinges_hist):
                for j in range(nh):
                    dth[k, j] = float(hinfs[j].get("dtheta", 0.0))
                    M[k, j] = float(hinfs[j].get("M", 0.0))
            theta = np.vstack([np.zeros((1, nh)), np.cumsum(dth, axis=0)])
            tt = t[:theta.shape[0]] if t.size >= theta.shape[0] else np.arange(theta.shape[0]) * float(last.get("dt", 1.0))

            beam_hinge_ids = [i for i, h in enumerate(model.hinges) if h.kind == "beam_shm"]
            for hid in beam_hinge_ids:
                fig, ax = plt.subplots()
                x = theta[:, hid]
                y = M[:, hid] / 1e3  # kNm
                lc = colored_line(x, y, tt)
                ax.add_collection(lc)
                ax.autoscale()
                ax.set_xlabel("θ_rel [rad]")
                ax.set_ylabel("M [kNm]")
                ax.set_title(f"Beam hinge H{hid}: M–θ (color=time)")
                ax.grid(True)
                cbar = fig.colorbar(lc, ax=ax)
                cbar.set_label("t [s]")
                add_snapshot(ax, snap)
                fig.tight_layout()
                fig.savefig(f"problem4_last_beam_hinge_H{hid}_Mtheta.png", dpi=150)
                plt.close(fig)
    except Exception:
        pass

    # --- N–M surface + hinge trajectories (columns) ---
    try:
        surface = meta.get("surface_col", None)
        if surface is not None:
            verts = np.array(surface.vertices, dtype=float)
            N = verts[:, 0] / 1e3
            Mp = verts[:, 1] / 1e3
            Mm = -Mp

            fig, ax = plt.subplots()
            ax.plot(N, Mp, linewidth=1.5, label="+M cap")
            ax.plot(N, Mm, linewidth=1.5, label="-M cap")

            hinges_hist = last.get("hinges", [])
            if isinstance(hinges_hist, list) and len(hinges_hist) > 2:
                nh = len(hinges_hist[0])
                Mhist = np.zeros((len(hinges_hist), nh), dtype=float)
                for k, hinfs in enumerate(hinges_hist):
                    for j in range(nh):
                        Mhist[k, j] = float(hinfs[j].get("M", 0.0))
                tt = t[:len(hinges_hist)]
                col_hinge_ids = [i for i, h in enumerate(model.hinges) if h.kind == "col_nm"]
                Nrefs = meta.get("N_ref_per_col_hinge", None)
                last_lc = None
                for idx_i, hid in enumerate(col_hinge_ids):
                    Nref = float(Nrefs[idx_i]) if isinstance(Nrefs, (list, tuple, np.ndarray)) and idx_i < len(Nrefs) else float(meta.get("N_ref", 0.0))
                    x = np.full_like(tt, Nref/1e3, dtype=float)
                    y = Mhist[:, hid] / 1e3
                    last_lc = colored_line(x, y, tt)
                    ax.add_collection(last_lc)
                    ax.text(x[0], y[0], f"H{hid}", fontsize=8, ha="left", va="bottom")
                ax.autoscale()
                if last_lc is not None:
                    cbar = fig.colorbar(last_lc, ax=ax)
                    cbar.set_label("t [s]")

            ax.set_xlabel("N [kN] (tension +)")
            ax.set_ylabel("M [kNm]")
            ax.set_title("N–M surface + column-hinge trajectories (color=time)")
            ax.grid(True)
            ax.legend(loc="best", fontsize=8)
            add_snapshot(ax, snap)
            fig.tight_layout()
            fig.savefig("problem4_last_NM_surface_and_paths.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass

    # --- Energy balance plots ---
    try:
        plot_energy_balance(last, model, drift_limit)
    except Exception:
        pass

def plot_ida(amps_g: np.ndarray, peak_drifts: List[float], drift_limit: float = 0.04):
    if len(peak_drifts) == 0:
        return
    fig, ax = plt.subplots()
    ax.plot(amps_g[:len(peak_drifts)], 100*np.array(peak_drifts), marker="o")
    ax.axhline(100*drift_limit, linestyle="--", label=f"drift limit = {100*drift_limit:.1f}%")
    ax.set_xlabel("A [g]")
    ax.set_ylabel("Peak drift [%]")
    ax.set_title("IDA: Peak drift vs amplitude")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig("problem4_IDA.png", dpi=150)
    plt.close(fig)

def main():
    drift_limit = 0.04
    alpha = -0.05

    peak_drifts, amps_used, last, model, meta = run_incremental_amplitudes(
        H=3.0, L=5.0, T0=0.5, zeta=0.02,
        drift_limit=drift_limit,
        amps_g=np.arange(0.1, 2.1, 0.1),
        t_end=10.0,
        base_dt=0.002,
        dt_min=0.00025,
        alpha=alpha,
    )

    # rebuild meta for plots (clean model object)
    model, meta = build_portal_beam_hinge(H=3.0, L=5.0, T0=0.5, zeta=0.02, P_gravity_total=1500e3)

    plot_ida(amps_used, peak_drifts, drift_limit=drift_limit)
    plot_results(last, model, meta, drift_limit=drift_limit)

    # dt sensitivity (use a low amplitude that actually ran)
    if amps_used.size > 0:
        A_sens = float(amps_used[min(1, amps_used.size - 1)])
    else:
        A_sens = 0.10

    print(f"\n--- dt sensitivity (A={A_sens:.2f}g) ---")
    dt_sensitivity_analysis(
        H=3.0, L=5.0, T0=0.5, zeta=0.02,
        A_g=A_sens,
        drift_limit=drift_limit,
        alpha=alpha,
        t_end=3.0,
        dts=(0.004, 0.002, 0.001, 0.0005),
    )

    if last is not None:
        print("Plots guardados como problem4_last_*.png + energy + dt_sensitivity")


if __name__ == "__main__":
    main()