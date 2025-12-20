"""Portal frame problem (Problema 4) using dc_solver core."""

from __future__ import annotations

import argparse

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from plastic_hinge import (
    RCSectionRect,
    RebarLayer,
    NMSurfacePolygon,
    PlasticHingeNM,
    ConcreteParabolicRect,
    SteelBilinearPerfect,
    Fiber2D,
    FiberSection2DStateful,
    rectangular_fiber_mesh,
)

from dc_solver.fem.nodes import Node, DofManager
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model
from dc_solver.fem.utils import discretize_member
from dc_solver.hinges.models import (
    ColumnHingeNM2D,
    HingeNM2DElement,
    SHMBeamHinge1D,
    FiberBeamHinge1D,
    FiberRotSpringElement,
    RotSpringElement,
    moment_capacity_from_polygon,
)
from dc_solver.integrators import solve_dynamic
from dc_solver.post.hinge_exports import export_problem4_hinges
from dc_solver.reporting.run_info import build_run_info, write_run_info
from dc_solver.post.hysteresis_gradient import add_time_gradient_line, add_colorbar
from dc_solver.post.plotting import plot_structure_states


def mirror_section_about_middepth(sec: RCSectionRect) -> RCSectionRect:
    layers = []
    for layer in sec.layers:
        y2 = sec.h - layer.y
        layers.append(RebarLayer(As=layer.As, y=y2))
    return RCSectionRect(b=sec.b, h=sec.h, fc=sec.fc, fy=sec.fy, Es=sec.Es,
                         layers=layers, n_fibers=sec.n_fibers)


def build_rc_rect_fiber_section_2d_stateful(
    *,
    b: float,
    h: float,
    cover: float,
    fc: float,
    fy: float,
    Es: float = 200e9,
    eps_c0: float = 0.002,
    eps_cu: float = 0.0035,
    ny: int = 50,
    nz: int = 30,
    clustering: str = "cosine",
    # Rebars described as (As_layer, y, n_bars)
    rebar_layers: list[tuple[float, float, int]] | None = None,
) -> FiberSection2DStateful:
    """Create a rectangular RC fiber section (2D mesh) with stateful steel.

    * Concrete: ConcreteParabolicRect (compression-only)
    * Steel: SteelBilinearPerfect parameters (fy,Es) but with stateful EP return mapping
      inside FiberSection2DStateful.

    Coordinates
    -----------
    y=0 at top fiber, y=h at bottom.
    z is across width, centered at z=0.
    """

    conc = ConcreteParabolicRect(fc=fc, eps_c0=eps_c0, eps_cu=eps_cu)
    steel = SteelBilinearPerfect(fy=fy, Es=Es)

    fibers: list[Fiber2D] = []
    fibers.extend(
        rectangular_fiber_mesh(
            b=b,
            h=h,
            ny=ny,
            nz=nz,
            mat=conc,
            clustering=clustering,
        )
    )

    if rebar_layers:
        for As_layer, y_layer, n_bars in rebar_layers:
            n_bars = max(1, int(n_bars))
            As_bar = float(As_layer) / float(n_bars)
            z_min = -0.5 * (b - 2.0 * cover)
            z_max = +0.5 * (b - 2.0 * cover)
            if n_bars == 1:
                z_pos = [0.0]
            else:
                z_pos = np.linspace(z_min, z_max, n_bars).tolist()
            for z in z_pos:
                fibers.append(Fiber2D(A=As_bar, y=float(y_layer), z=float(z), mat=steel))

    sec = FiberSection2DStateful(fibers=fibers, y_c=0.5 * h, z_c=0.0)
    return sec


def build_nm_surface(sec: RCSectionRect, npts: int = 90, tension_positive: bool = True) -> NMSurfacePolygon:
    pts1 = sec.sample_interaction_curve(n=npts)
    pts2 = mirror_section_about_middepth(sec).sample_interaction_curve(n=npts)
    pts = np.vstack([pts1, pts2, pts1 * np.array([1.0, -1.0]), pts2 * np.array([1.0, -1.0])])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if tension_positive:
        pts = pts.copy()
        pts[:, 0] *= -1.0
    return NMSurfacePolygon.from_points(pts)


def _outputs_dir(subdir: Optional[str] = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if subdir:
        out = out / str(subdir)
        out.mkdir(parents=True, exist_ok=True)
    return out


def _rebar_area(phi_m: float) -> float:
    return math.pi * (phi_m / 2.0) ** 2


def build_portal_beam_hinge(
    H: float = 3.0,
    L: float = 5.0,
    T0: float = 0.5,
    zeta: float = 0.02,
    P_gravity_total: float = 1500e3,
    nseg: int = 6,
    cover: float = 0.05,
    nlgeom: bool = False,
    beam_hinge: str = "shm",  # "shm" | "fiber"
) -> Tuple[Model, Dict]:
    dm = DofManager()

    n0 = Node(0.0, 0.0, dm.new_trans(), dm.new_rot())
    n1 = Node(L, 0.0, dm.new_trans(), dm.new_rot())
    n2 = Node(0.0, H, dm.new_trans(), dm.new_rot())
    n3 = Node(L, H, dm.new_trans(), dm.new_rot())

    nodes: List[Node] = [n0, n1, n2, n3]

    def aux_at(j: int) -> int:
        nj = nodes[j]
        na = Node(nj.x, nj.y, nj.dof_u, dm.new_rot())
        nodes.append(na)
        return len(nodes) - 1

    # Aux nodes:
    # - Column bases: separate rotation DOF so base hinges act between (base node) and (column bottom).
    # - Beam ends: separate rotation DOF so beam end hinges act between (joint node) and (beam end).
    i0L = aux_at(0)
    i1R = aux_at(1)
    i2B = aux_at(2)
    i3B = aux_at(3)

    fc = 30e6
    fy = 420e6
    Es = 200e9

    b_col, h_col = 0.40, 0.60
    phi20 = 20e-3
    layers_col = [
        RebarLayer(As=4 * _rebar_area(phi20), y=cover),
        RebarLayer(As=4 * _rebar_area(phi20), y=h_col - cover),
    ]
    sec_col = RCSectionRect(b=b_col, h=h_col, fc=fc, fy=fy, Es=Es, layers=layers_col, n_fibers=80)
    surf_col = build_nm_surface(sec_col, npts=80, tension_positive=True)

    b_beam, h_beam = 0.25, 0.50
    layers_beam = [
        RebarLayer(As=3 * _rebar_area(phi20), y=cover),
        RebarLayer(As=2 * _rebar_area(phi20), y=h_beam - cover),
    ]
    sec_beam = RCSectionRect(b=b_beam, h=h_beam, fc=fc, fy=fy, Es=Es, layers=layers_beam, n_fibers=80)
    surf_beam = build_nm_surface(sec_beam, npts=80, tension_positive=True)
    My_beam = moment_capacity_from_polygon(surf_beam, N=0.0)
    A_beam = b_beam * h_beam
    I_beam = b_beam * (h_beam ** 3) / 12.0

    A_col = b_col * h_col
    I_col = b_col * (h_col ** 3) / 12.0

    E = 30e9

    nseg_use = int(nseg)
    # IMPORTANT: columns connect to the *joint* nodes (2,3) so joint rotations are shared.
    # Only the beam end is decoupled via the rotational spring (2 -> i2B and 3 -> i3B).
    left_nodes = discretize_member(i0L, 2, nseg_use, nodes, dm)
    right_nodes = discretize_member(i1R, 3, nseg_use, nodes, dm)
    beam_nodes = discretize_member(i2B, i3B, nseg_use, nodes, dm)

    beams: List[FrameElementLinear2D] = []

    def add_member(node_ids: List[int], A: float, I: float) -> List[int]:
        elem_ids: List[int] = []
        for a, b in zip(node_ids[:-1], node_ids[1:]):
            beams.append(FrameElementLinear2D(a, b, E=E, A=A, I=I, nodes=nodes))
            elem_ids.append(len(beams) - 1)
        return elem_ids

    left_elems = add_member(left_nodes, A_col, I_col)
    right_elems = add_member(right_nodes, A_col, I_col)
    beam_elems = add_member(beam_nodes, A_beam, I_beam)

    Lp_col = 0.5 * h_col
    k_col0 = 6.0 * E * I_col / H
    k_beam0 = 6.0 * E * I_beam / L
    KN_col = E * A_col / Lp_col
    KM_col = E * I_col / Lp_col

    hinges: List[RotSpringElement | FiberRotSpringElement | HingeNM2DElement] = []
    hinge_left = ColumnHingeNM2D(
        hinge=PlasticHingeNM(surface=surf_col, K=np.diag([KN_col, KM_col]), enable_substepping=True),
    )
    hinge_right = ColumnHingeNM2D(
        hinge=PlasticHingeNM(surface=surf_col, K=np.diag([KN_col, KM_col]), enable_substepping=True),
    )
    hinges.append(HingeNM2DElement(0, i0L, hinge_left, nodes))
    hinges.append(HingeNM2DElement(1, i1R, hinge_right, nodes))

    beam_hinge = str(beam_hinge).lower().strip()
    if beam_hinge == "shm":
        shm_left = SHMBeamHinge1D(K0_0=k_beam0, My_0=My_beam)
        shm_right = SHMBeamHinge1D(K0_0=k_beam0, My_0=My_beam)
        hinges.append(RotSpringElement(2, i2B, "beam_shm", None, shm_left, nodes))
        hinges.append(RotSpringElement(3, i3B, "beam_shm", None, shm_right, nodes))
    elif beam_hinge == "fiber":
        # Fiber hinge: same materials as above but with stateful steel (better unloading / residual strains)
        beam_sec = build_rc_rect_fiber_section_2d_stateful(
            b=b_beam,
            h=h_beam,
            cover=cover,
            fc=fc,
            fy=fy,
            Es=Es,
            ny=50,
            nz=30,
            clustering="cosine",
            rebar_layers=[
                (3 * _rebar_area(phi20), cover, 3),
                (2 * _rebar_area(phi20), h_beam - cover, 2),
            ],
        )
        # Separate sections for left/right so the state does not leak between hinges.
        beam_sec_R = build_rc_rect_fiber_section_2d_stateful(
            b=b_beam,
            h=h_beam,
            cover=cover,
            fc=fc,
            fy=fy,
            Es=Es,
            ny=50,
            nz=30,
            clustering="cosine",
            rebar_layers=[
                (3 * _rebar_area(phi20), cover, 3),
                (2 * _rebar_area(phi20), h_beam - cover, 2),
            ],
        )

        Lp_beam = 0.5 * h_beam
        fiber_left = FiberBeamHinge1D(section=beam_sec, Lp=Lp_beam, N_target=0.0)
        fiber_right = FiberBeamHinge1D(section=beam_sec_R, Lp=Lp_beam, N_target=0.0)
        hinges.append(FiberRotSpringElement(2, i2B, fiber_left, nodes))
        hinges.append(FiberRotSpringElement(3, i3B, fiber_right, nodes))
    else:
        raise ValueError(f"Unknown beam_hinge: {beam_hinge!r} (use 'shm' or 'fiber')")

    fixed = np.array([
        nodes[0].dof_u[0], nodes[0].dof_u[1], nodes[0].dof_th,
        nodes[1].dof_u[0], nodes[1].dof_u[1], nodes[1].dof_th,
    ], dtype=int)

    nd = dm.ndof
    mass = np.zeros(nd)
    C = np.zeros(nd)
    p0 = np.zeros(nd)

    total_length = sum(float(e._geom()[0]) for e in beams)
    if total_length > 0.0:
        w = float(P_gravity_total) / total_length
        for e in beams:
            f_g = e.equiv_nodal_load_global((0.0, -w))
            dofs = e.dofs()
            for a, ia in enumerate(dofs):
                p0[ia] += f_g[a]

    model = Model(
        nodes=nodes,
        beams=beams,
        hinges=hinges,
        fixed_dofs=fixed,
        mass_diag=mass,
        C_diag=C,
        load_const=p0,
        col_hinge_groups=[],
        nlgeom=nlgeom,
    )

    K_story = story_stiffness_linear(model, top_nodes=(2, 3))
    omega0 = 2.0 * math.pi / T0
    M_total = K_story / (omega0 ** 2)

    mass[nodes[2].dof_u[0]] = 0.5 * M_total
    mass[nodes[3].dof_u[0]] = 0.5 * M_total
    C[:] = 2.0 * zeta * omega0 * mass

    meta = {
        "K_story": K_story,
        "M_total": M_total,
        "T0": T0,
        "omega0": omega0,
        "section_col": sec_col,
        "surface_col": surf_col,
        "section_beam": sec_beam,
        "surface_beam": surf_beam,
        "My_beam": My_beam,
        "H": H,
        "L": L,
        "nseg": nseg_use,
        "member_elements": {"left": left_elems, "right": right_elems, "beam": beam_elems},
    }
    return model, meta


def story_stiffness_linear(model: Model, top_nodes: Tuple[int, int]) -> float:
    nd = model.ndof()
    fd = model.free_dofs()

    u_comm = np.zeros(nd)
    u_trial = np.zeros(nd)

    model.update_column_yields(u_comm)
    K, _, _ = model.assemble(u_trial, u_comm)

    f = np.zeros(nd)
    ux2 = model.nodes[top_nodes[0]].dof_u[0]
    ux3 = model.nodes[top_nodes[1]].dof_u[0]
    f[ux2] = 0.5
    f[ux3] = 0.5
    f_free = f[fd]

    u_free = np.linalg.solve(K + 1e-14 * np.eye(fd.size), f_free)
    u = np.zeros(nd)
    u[fd] = u_free
    u_top = 0.5 * (u[ux2] + u[ux3])
    return float(1.0 / u_top)


def make_time(t_end: float, dt: float) -> np.ndarray:
    return np.arange(0.0, t_end + 1e-12, dt)


def ag_fun(t: np.ndarray, A: float) -> np.ndarray:
    return A * np.cos(0.2 * np.pi * t) * np.sin(4.0 * np.pi * t)


def export_hinge_hysteresis_gradient(
    out: Path,
    model: Model,
    last: Dict,
    *,
    max_hinges: int = 10,
) -> None:
    """Export M–Δθ hysteresis for hinges with a time color gradient.

    The dynamic solvers store a per-step hinge info list in last["hinges"].
    We reconstruct Δθ(t) from the global displacement history u(t).
    """
    hinge_hist = last.get("hinges", [])
    if not hinge_hist:
        return

    u = np.asarray(last["u"], dtype=float)
    t = np.asarray(last["t"], dtype=float)

    n = min(len(hinge_hist), u.shape[0] - 1, t.shape[0] - 1)
    if n <= 1:
        return

    time = t[1 : n + 1]

    hinges = getattr(model, "hinges", [])
    if not hinges:
        return

    # Determine number of hinge entries (guard against mismatches)
    n_hinges = min(len(hinges), len(hinge_hist[0]))
    if n_hinges <= 0:
        return

    # Rank hinges by moment range to avoid exporting dozens of near-flat plots
    ranges = []
    for ih in range(n_hinges):
        M = np.array([float(hinge_hist[k][ih].get("M", float("nan"))) for k in range(n)], dtype=float)
        rng = float(np.nanmax(M) - np.nanmin(M)) if np.isfinite(M).any() else 0.0
        ranges.append(rng)

    selected = sorted(range(n_hinges), key=lambda i: -ranges[i])[: max_hinges]
    selected = [ih for ih in selected if ranges[ih] > 0.0] or selected[:1]

    norm = Normalize(vmin=float(time.min()), vmax=float(time.max()))

    # Combined figure
    fig, axes = plt.subplots(len(selected), 1, figsize=(7.2, 2.4 * len(selected)), squeeze=False)
    axes = axes[:, 0]

    extras_keys = ["a", "N", "N_res", "eps0", "kappa", "iters", "active", "My"]

    last_lc = None
    for ax, ih in zip(axes, selected):
        dofs = np.asarray(hinges[ih].dofs(), dtype=int)
        th_i = u[1 : n + 1, dofs[2]]
        th_j = u[1 : n + 1, dofs[5]]
        dth = th_j - th_i
        M = np.array([float(hinge_hist[k][ih].get("M", float("nan"))) for k in range(n)], dtype=float)

        last_lc = add_time_gradient_line(ax, dth, M, c=time, norm=norm, lw=2.0)
        kind = getattr(hinges[ih], "kind", "hinge")
        ni = getattr(hinges[ih], "ni", "?")
        nj = getattr(hinges[ih], "nj", "?")
        ax.set_ylabel(f"M [N-m]\n#{ih} {kind} ({ni}-{nj})")
        ax.grid(True, alpha=0.3)

        # CSV export for this hinge (include optional debug fields if present)
        cols = [time, dth, M]
        headers = ["t", "dtheta", "M"]
        for k in extras_keys:
            arr = np.array([hinge_hist[kk][ih].get(k, float("nan")) for kk in range(n)], dtype=float)
            cols.append(arr)
            headers.append(k)
        csv = np.column_stack(cols)
        np.savetxt(
            out / f"problem4_hinge_{ih}_{kind}_hysteresis.csv",
            csv,
            delimiter=",",
            header=",".join(headers),
            comments="",
        )

    axes[-1].set_xlabel("Δθ [rad]")
    if last_lc is not None:
        add_colorbar(last_lc, axes[-1], label="t [s]")

    fig.suptitle("Problem 4 — hinge hysteresis (time gradient)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "problem4_hinge_hysteresis_gradient.png", dpi=170)
    plt.close(fig)

    # Individual hinge figures (selected only)
    for ih in selected:
        dofs = np.asarray(hinges[ih].dofs(), dtype=int)
        th_i = u[1 : n + 1, dofs[2]]
        th_j = u[1 : n + 1, dofs[5]]
        dth = th_j - th_i
        M = np.array([float(hinge_hist[k][ih].get("M", float("nan"))) for k in range(n)], dtype=float)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        lc = add_time_gradient_line(ax, dth, M, c=time, norm=norm, lw=2.2)
        add_colorbar(lc, ax, label="t [s]")
        kind = getattr(hinges[ih], "kind", "hinge")
        ni = getattr(hinges[ih], "ni", "?")
        nj = getattr(hinges[ih], "nj", "?")
        ax.set_title(f"Hinge #{ih} {kind} ({ni}-{nj})")
        ax.set_xlabel("Δθ [rad]")
        ax.set_ylabel("M [N-m]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"problem4_hinge_{ih}_{kind}_hysteresis_gradient.png", dpi=170)
        plt.close(fig)


def run_incremental_amplitudes(
    integrator: str = "hht",
    H=3.0,
    L=5.0,
    T0=0.5,
    zeta=0.02,
    drift_limit=0.10,
    snapshot_limit=0.04,
    amps_g=np.arange(0.1, 2.1, 0.1),
    t_end=10.0,
    base_dt=0.002,
    dt_min=0.00025,
    alpha=-0.05,
    nseg: int = 6,
    cover: float = 0.05,
    nlgeom: bool = False,
    beam_hinge: str = "shm",
):
    g = 9.81
    model, meta = build_portal_beam_hinge(
        H=H,
        L=L,
        T0=T0,
        zeta=zeta,
        P_gravity_total=1500e3,
        nseg=nseg,
        cover=cover,
        nlgeom=nlgeom,
        beam_hinge=beam_hinge,
    )

    peak_drifts = []
    dt_hist = []
    last = None

    for A_g in amps_g:
        A = float(A_g) * g
        dt = base_dt
        while True:
            t = make_time(t_end, dt)
            ag = ag_fun(t, A)
            try:
                out = solve_dynamic(
                    integrator,
                    model=model,
                    t=t,
                    ag=ag,
                    drift_height=H,
                    drift_limit=drift_limit,
                    drift_snapshot=snapshot_limit,
                    alpha=alpha,
                    beta=0.25,
                    gamma=0.50,
                    base_nodes=(0, 1),
                    drift_nodes=(2, 3),
                    max_iter=50,
                    tol=1e-6,
                    verbose=False,
                )
                out["A_ms2"] = float(A)
                out["A_input_g"] = float(A_g)
                out["zeta"] = float(zeta)
                out["T0"] = float(T0)
                last = out
                dt_hist.append(float(out["dt"]))
                break
            except RuntimeError:
                if dt <= dt_min + 1e-15:
                    meta.update({"dt_hist": dt_hist, "amps_g_used": amps_g[:len(peak_drifts)]})
                    return peak_drifts, amps_g[:len(peak_drifts)], last, model, meta
                dt *= 0.5

        pk = float(np.max(np.abs(out["drift"])))
        peak_drifts.append(pk)
        if pk >= drift_limit:
            break

    meta.update({"dt_hist": dt_hist, "amps_g_used": amps_g[:len(peak_drifts)]})
    return peak_drifts, amps_g[:len(peak_drifts)], last, model, meta


def plot_results(
    last: Dict[str, np.ndarray],
    model: Model,
    meta: Dict,
    drift_limit: float = 0.10,
    snapshot_limit: Optional[float] = None,
    outdir: Optional[Path] = None,
) -> None:
    if last is None:
        return
    out = _outputs_dir() if outdir is None else Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    H = float(meta.get("H", 1.0))
    plot_structure_states(
        model,
        last,
        drift_height=H,
        snapshot_limit=snapshot_limit,
        outfile=str(out / "problem4_states_U.png"),
        field="U",
        shared_colorbar=True,
    )
    plot_structure_states(
        model,
        last,
        drift_height=H,
        snapshot_limit=snapshot_limit,
        outfile=str(out / "problem4_states_S.png"),
        field="S",
        shared_colorbar=True,
    )
    plot_structure_states(
        model,
        last,
        drift_height=H,
        snapshot_limit=snapshot_limit,
        outfile=str(out / "problem4_states_members.png"),
        field="both",
    )

    # Full exports for dissertation figures (CSV + plots)
    export_problem4_hinges(out, model, last)


def main():

    parser = argparse.ArgumentParser(description="Problema 4: pórtico con rótulas (time-history).")
    parser.add_argument("--integrator", default="hht", choices=["hht", "newmark", "explicit"], help="Time integrator.")
    parser.add_argument("--nlgeom", action="store_true", help="Enable geometric nonlinearity (P-Delta).")
    parser.add_argument("--nseg", type=int, default=6, help="Segments per member (visualization/mesh).")
    parser.add_argument(
        "--beam-hinge",
        default="shm",
        choices=["shm", "fiber", "compare"],
        help="Beam end hinge model: 'shm', 'fiber', or 'compare' (runs both).",
    )
    args = parser.parse_args()

    drift_limit = 0.10
    snapshot_limit = 0.04
    alpha = -0.05
    nseg = int(args.nseg)
    nlgeom = bool(args.nlgeom)

    def _write_basic_plots(out: Path, last: Optional[Dict[str, np.ndarray]]) -> None:
        if last is None:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(last["t"], last["drift"], label="drift")
        ax.axhline(drift_limit, color="r", linestyle="--", label="limit")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Drift")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "problem4_drift_time.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(last["drift"], last["Vb"])
        ax.set_xlabel("Drift")
        ax.set_ylabel("Base shear [N]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "problem4_vb_drift.png", dpi=160)
        plt.close(fig)

    def _write_summary(out: Path, meta: Dict, amps_used: np.ndarray, peak_drifts: list[float]) -> None:
        lines = [
            "Problem 4 summary",
            f"beam_hinge={args.beam_hinge}",
            f"integrator={args.integrator}",
            f"drift_limit={drift_limit:.3f}",
            f"snapshot_limit={snapshot_limit:.3f}",
            f"alpha={alpha:.3f}",
            "A_g,peak_drift,dt",
        ]
        dt_hist = meta.get("dt_hist", [])
        for i, (amp, drift) in enumerate(zip(amps_used, peak_drifts)):
            dt_val = dt_hist[i] if i < len(dt_hist) else float("nan")
            lines.append(f"{amp:.3f},{drift:.6f},{dt_val:.6f}")
        if amps_used.size:
            lines.append(f"collapse_A_g={amps_used[-1]:.3f}")
        (out / "problem4_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    def _write_runinfo(out: Path, meta: Dict, last: Optional[Dict[str, np.ndarray]]) -> None:
        ri_meta: Dict[str, object] = {
            "problem": 4,
            "integrator": str(args.integrator),
            "beam_hinge": str(args.beam_hinge),
            "nlgeom": bool(nlgeom),
            "nseg": int(nseg),
            "drift_limit": float(drift_limit),
            "snapshot_limit": float(snapshot_limit),
            "alpha": float(alpha),
            "dt_hist": meta.get("dt_hist", []),
            "amps_g_used": meta.get("amps_g_used", []),
        }
        if last is not None:
            try:
                ri_meta.update(
                    {
                        "dt": float(last.get("dt", float("nan"))),
                        "n_steps": int(len(last.get("t", [])) - 1),
                        "peak_drift": float(np.max(np.abs(last.get("drift", np.array([0.0]))))),
                    }
                )
            except Exception:
                pass
        info = build_run_info(job="problem4_portico", output_dir=str(out), meta=ri_meta)
        write_run_info(out, base_name="problem4_runinfo", info=info)

    if args.beam_hinge != "compare":
        out = _outputs_dir(f"problem4_{args.beam_hinge}_{args.integrator}")
        peak_drifts, amps_used, last, model, meta = run_incremental_amplitudes(
            integrator=args.integrator,
            H=3.0,
            L=5.0,
            T0=0.5,
            zeta=0.02,
            drift_limit=drift_limit,
            amps_g=np.arange(0.1, 2.1, 0.1),
            t_end=10.0,
            base_dt=0.002,
            dt_min=0.00025,
            alpha=alpha,
            nseg=nseg,
            nlgeom=nlgeom,
            beam_hinge=args.beam_hinge,
        )
        plot_results(last, model, meta, drift_limit=drift_limit, snapshot_limit=snapshot_limit, outdir=out)
        _write_basic_plots(out, last)
        _write_summary(out, meta, amps_used, peak_drifts)
        _write_runinfo(out, meta, last)
        return

    # compare mode
    out_shm = _outputs_dir(f"problem4_shm_{args.integrator}")
    out_fib = _outputs_dir(f"problem4_fiber_{args.integrator}")
    out_cmp = _outputs_dir(f"problem4_compare_{args.integrator}")

    pd_shm, amps_shm, last_shm, model_shm, meta_shm = run_incremental_amplitudes(
        integrator=args.integrator,
        H=3.0,
        L=5.0,
        T0=0.5,
        zeta=0.02,
        drift_limit=drift_limit,
        amps_g=np.arange(0.1, 2.1, 0.1),
        t_end=10.0,
        base_dt=0.002,
        dt_min=0.00025,
        alpha=alpha,
        nseg=nseg,
        nlgeom=nlgeom,
        beam_hinge="shm",
    )
    plot_results(last_shm, model_shm, meta_shm, drift_limit=drift_limit, snapshot_limit=snapshot_limit, outdir=out_shm)
    _write_basic_plots(out_shm, last_shm)
    _write_summary(out_shm, meta_shm, amps_shm, pd_shm)
    _write_runinfo(out_shm, meta_shm, last_shm)

    pd_fib, amps_fib, last_fib, model_fib, meta_fib = run_incremental_amplitudes(
        integrator=args.integrator,
        H=3.0,
        L=5.0,
        T0=0.5,
        zeta=0.02,
        drift_limit=drift_limit,
        amps_g=np.arange(0.1, 2.1, 0.1),
        t_end=10.0,
        base_dt=0.002,
        dt_min=0.00025,
        alpha=alpha,
        nseg=nseg,
        nlgeom=nlgeom,
        beam_hinge="fiber",
    )
    plot_results(last_fib, model_fib, meta_fib, drift_limit=drift_limit, snapshot_limit=snapshot_limit, outdir=out_fib)
    _write_basic_plots(out_fib, last_fib)
    _write_summary(out_fib, meta_fib, amps_fib, pd_fib)
    _write_runinfo(out_fib, meta_fib, last_fib)

    # compare summary + overlay
    common_n = min(len(amps_shm), len(amps_fib))
    amps = amps_shm[:common_n]
    y_shm = np.array(pd_shm[:common_n], dtype=float)
    y_fib = np.array(pd_fib[:common_n], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(amps, y_shm, marker="o", label="SHM beam hinge")
    ax.plot(amps, y_fib, marker="s", label="Fiber beam hinge")
    ax.axhline(drift_limit, color="r", linestyle="--", label="limit")
    ax.set_xlabel("A_g [g]")
    ax.set_ylabel("Peak drift")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_cmp / "problem4_compare_peakdrift.png", dpi=170)
    plt.close(fig)

    if last_shm is not None and last_fib is not None:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(last_shm["drift"], last_shm["Vb"], label="SHM")
        ax.plot(last_fib["drift"], last_fib["Vb"], label="Fiber")
        ax.set_xlabel("Drift")
        ax.set_ylabel("Base shear [N]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_cmp / "problem4_compare_vb_drift.png", dpi=170)
        plt.close(fig)

    # CSV summary
    dt_shm = meta_shm.get("dt_hist", [])
    dt_fib = meta_fib.get("dt_hist", [])
    lines = ["A_g,peak_drift_shm,peak_drift_fiber,dt_shm,dt_fiber"]
    for i in range(common_n):
        dts = dt_shm[i] if i < len(dt_shm) else float("nan")
        dtf = dt_fib[i] if i < len(dt_fib) else float("nan")
        lines.append(f"{amps[i]:.3f},{y_shm[i]:.6f},{y_fib[i]:.6f},{dts:.6f},{dtf:.6f}")
    (out_cmp / "problem4_compare_summary.csv").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
