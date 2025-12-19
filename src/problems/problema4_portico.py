"""Portal frame problem (Problema 4) using dc_solver core."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plastic_hinge import RCSectionRect, RebarLayer, NMSurfacePolygon

from dc_solver.fem.nodes import Node, DofManager
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model
from dc_solver.fem.utils import discretize_member
from dc_solver.hinges.models import (
    ColumnHingeNMRot,
    SHMBeamHinge1D,
    RotSpringElement,
    moment_capacity_from_polygon,
)
from dc_solver.integrators.hht_alpha import hht_alpha_newton
from dc_solver.post.plotting import plot_structure_states


def mirror_section_about_middepth(sec: RCSectionRect) -> RCSectionRect:
    layers = []
    for layer in sec.layers:
        y2 = sec.h - layer.y
        layers.append(RebarLayer(As=layer.As, y=y2))
    return RCSectionRect(b=sec.b, h=sec.h, fc=sec.fc, fy=sec.fy, Es=sec.Es,
                         layers=layers, n_fibers=sec.n_fibers)


def build_nm_surface(sec: RCSectionRect, npts: int = 90, tension_positive: bool = True) -> NMSurfacePolygon:
    pts1 = sec.sample_interaction_curve(n=npts)
    pts2 = mirror_section_about_middepth(sec).sample_interaction_curve(n=npts)
    pts = np.vstack([pts1, pts2, pts1 * np.array([1.0, -1.0]), pts2 * np.array([1.0, -1.0])])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if tension_positive:
        pts = pts.copy()
        pts[:, 0] *= -1.0
    return NMSurfacePolygon.from_points(pts)


def _outputs_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
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

    i0L = aux_at(0)
    i2L = aux_at(2)
    i1R = aux_at(1)
    i3R = aux_at(3)
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
    left_nodes = discretize_member(i0L, i2L, nseg_use, nodes, dm)
    right_nodes = discretize_member(i1R, i3R, nseg_use, nodes, dm)
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

    k_col0 = 6.0 * E * I_col / H
    k_beam0 = 6.0 * E * I_beam / L

    hinges: List[RotSpringElement] = []
    hinges.append(RotSpringElement(0, i0L, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    hinges.append(RotSpringElement(2, i2L, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    hinges.append(RotSpringElement(1, i1R, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    hinges.append(RotSpringElement(3, i3R, "col_nm", ColumnHingeNMRot(surface=surf_col, k0=k_col0), None, nodes))
    shm_left = SHMBeamHinge1D(K0_0=k_beam0, My_0=My_beam)
    shm_right = SHMBeamHinge1D(K0_0=k_beam0, My_0=My_beam)
    hinges.append(RotSpringElement(2, i2B, "beam_shm", None, shm_left, nodes))
    hinges.append(RotSpringElement(3, i3B, "beam_shm", None, shm_right, nodes))

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
        col_hinge_groups=[
            (0, left_elems[0], +1), (1, left_elems[-1], +1),
            (2, right_elems[0], +1), (3, right_elems[-1], +1),
        ],
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


def run_incremental_amplitudes(
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
                out = hht_alpha_newton(
                    model,
                    t,
                    ag,
                    drift_height=H,
                    drift_limit=drift_limit,
                    drift_snapshot=snapshot_limit,
                    alpha=alpha,
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
) -> None:
    if last is None:
        return
    out = _outputs_dir()
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


def main():
    drift_limit = 0.10
    snapshot_limit = 0.04
    alpha = -0.05
    nseg = 6
    nlgeom = False

    peak_drifts, amps_used, last, model, meta = run_incremental_amplitudes(
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
    )
    plot_results(last, model, meta, drift_limit=drift_limit, snapshot_limit=snapshot_limit)

    out = _outputs_dir()
    if last is not None:
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

    lines = [
        "Problem 4 summary",
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


if __name__ == "__main__":
    main()
