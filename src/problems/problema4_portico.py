"""Portal frame problem (Problema 4) using dc_solver core."""

from __future__ import annotations

import argparse
from dc_solver.utils.gravity import solve_gravity_only

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
    ColumnHingeNMRot,
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
from dc_solver.post.plotting import plot_structure_states, write_member_stress_csv
from dc_solver.post.energy_balance import export_anregung, export_energy_balance


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


def distribute_lumped_mass_from_beams(
    *,
    nodes: List[Node],
    beams: List[FrameElementLinear2D],
    mass: np.ndarray,
    M_total: float,
    include_rot_inertia: bool = True,
) -> None:
    """Distribute a target total physical mass along all frame members.

    The solver uses a diagonal (lumped) mass vector. We approximate a consistent
    member mass by lumping half of each element's mass to each end node.

    - Translational mass is assigned to BOTH ux and uy DOFs at each node.
    - Optional rotational inertia is assigned to theta DOFs using a simple beam-like
      estimate per element end: I_end ≈ (m' * L^3) / 24, with m' = M_total / sum(L).
    """
    total_L = sum(float(e._geom()[0]) for e in beams)
    if total_L <= 0.0:
        return
    if M_total <= 0.0:
        return

    m_per_L = float(M_total) / float(total_L)

    for e in beams:
        L_e = float(e._geom()[0])
        if not np.isfinite(L_e) or L_e <= 0.0:
            continue

        m_end = 0.5 * m_per_L * L_e  # kg
        for n_id in (int(e.ni), int(e.nj)):
            ux, uy = nodes[n_id].dof_u
            mass[ux] += m_end
            mass[uy] += m_end
            if include_rot_inertia:
                th = nodes[n_id].dof_th
                I_end = m_per_L * (L_e ** 3) / 24.0  # kg*m^2
                mass[th] += I_end



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
    mass_mode: str = "distributed",  # "roof" | "distributed"
    include_rot_inertia: bool = True,
    fiber_ny: int = 20,
    fiber_nz: int = 14,
    fiber_line_search: bool = False,
    out_dir: Path | None = None,
    explicit_mass_dt: Optional[float] = None,
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



    # --- Plastic hinge lengths (macro-element end zones) ---
    # Literature practice (e.g., Cáceres / De la Llera): Lp is of order section depth.
    Lp_col = float(min(b_col, h_col))
    Lp_beam = float(min(b_beam, h_beam))

    # --- Option 1 (common): shortened elastic member (rigid end zones / end offsets) ---
    # Keep geometric length for kinematics and loading, but scale A and I of the elastic segments so that
    # EA/L and EI/L^3 match an effective elastic length Le = L_full - (Lp_i + Lp_j).
    def _scale_props_for_shortened_elastic(A: float, I: float, L_full: float, Lp_i: float, Lp_j: float) -> tuple[float, float, float, float]:
        Le = float(L_full) - float(Lp_i) - float(Lp_j)
        Le = max(Le, 1e-9)
        s = float(L_full) / Le
        A_eff = float(A) * s
        I_eff = float(I) * (s ** 3)
        return A_eff, I_eff, Le, s

    # Column: hinge zone at base only -> Le = H - Lp_col
    A_col_eff, I_col_eff, Le_col, s_col = _scale_props_for_shortened_elastic(A_col, I_col, H, Lp_col, 0.0)
    # Beam: hinge zones at both ends -> Le = L - 2*Lp_beam
    A_beam_eff, I_beam_eff, Le_beam, s_beam = _scale_props_for_shortened_elastic(A_beam, I_beam, L, Lp_beam, Lp_beam)

    # Initial elastic rotational stiffness of beam hinges (Cáceres: kθ = 2EI/Lp)
    k_beam0 = 2.0 * E * I_beam / Lp_beam

    # Column hinge local elastic stiffnesses (Cáceres: kδ = EA/Lp, kθ = 2EI/Lp)
    KN_col = E * A_col / Lp_col
    KM_col = 2.0 * E * I_col / Lp_col

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

    left_elems = add_member(left_nodes, A_col_eff, I_col_eff)
    right_elems = add_member(right_nodes, A_col_eff, I_col_eff)
    beam_elems = add_member(beam_nodes, A_beam_eff, I_beam_eff)
    hinges: List[RotSpringElement | FiberRotSpringElement | HingeNM2DElement] = []

    # --- Column plastic hinges (N-M interaction -> My(N), rotation-only spring) ---
    col_left = ColumnHingeNMRot(surface=surf_col, k0=KM_col, alpha_post=1e-4)
    col_right = ColumnHingeNMRot(surface=surf_col, k0=KM_col, alpha_post=1e-4)
    # Initialize with N_ref=0; will be updated every increment via Model.update_column_yields(...)
    col_left.set_yield_from_N(0.0)
    col_right.set_yield_from_N(0.0)
    idx_col_L = len(hinges)
    hinges.append(RotSpringElement(0, i0L, "col_nm", col_left, None, nodes))
    idx_col_R = len(hinges)
    hinges.append(RotSpringElement(1, i1R, "col_nm", col_right, None, nodes))


    beam_hinge = str(beam_hinge).lower().strip()
    if beam_hinge == "shm":
        # SHM calibration (v9): mild degradation + My(N) under compression
        Ncr_beam = float(A_beam) * float(fy)
        shm_left = SHMBeamHinge1D(
            K0_0=k_beam0,
            My_0=My_beam,
            alpha_post=0.03,
            bw_n=10.0,
            pinch=0.0,
            b1=0.05,
            b2=0.15,
            eta=1.0,
            N_cr=Ncr_beam,
            My_floor_frac_N=0.6,
        )
        shm_right = SHMBeamHinge1D(
            K0_0=k_beam0,
            My_0=My_beam,
            alpha_post=0.03,
            bw_n=10.0,
            pinch=0.0,
            b1=0.05,
            b2=0.15,
            eta=1.0,
            N_cr=Ncr_beam,
            My_floor_frac_N=0.6,
        )

        # Couple each hinge to its adjacent roof-beam element so Model.assemble can update N_comp_current.
        hinges.append(
            RotSpringElement(
                2,
                i2B,
                "beam_shm",
                None,
                shm_left,
                nodes,
                beam_idx=int(beam_elems[0]),
                beam_sign=-1.0,
                name="beam_shm_L",
            )
        )
        hinges.append(
            RotSpringElement(
                3,
                i3B,
                "beam_shm",
                None,
                shm_right,
                nodes,
                beam_idx=int(beam_elems[-1]),
                beam_sign=-1.0,
                name="beam_shm_R",
            )
        )
    elif beam_hinge == "fiber":
        # Fiber hinge: same materials as above but with stateful steel (better unloading / residual strains)
        beam_sec = build_rc_rect_fiber_section_2d_stateful(
            b=b_beam,
            h=h_beam,
            cover=cover,
            fc=fc,
            fy=fy,
            Es=Es,
            ny=int(fiber_ny), nz=int(fiber_nz),
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
            ny=int(fiber_ny), nz=int(fiber_nz),
            clustering="cosine",
            rebar_layers=[
                (3 * _rebar_area(phi20), cover, 3),
                (2 * _rebar_area(phi20), h_beam - cover, 2),
            ],
        )
        # Use the same Lp_beam defined above (macro-element hinge length)
        # NOTE: N_target should follow the *actual* axial force in the roof beam.
        # We attach each fiber spring to the adjacent roof-beam frame element so that
        # Model.assemble can update hinge.N_target every iteration/time step.
        fiber_left = FiberBeamHinge1D(
            section=beam_sec,
            Lp=Lp_beam,
            N_target=0.0,
            kappa_factor=2.0,
            moment_sign=-1.0,
            tol_N=100.0,   
            max_iter_eps0=80,
            line_search=bool(fiber_line_search),
        )
        fiber_right = FiberBeamHinge1D(
            section=beam_sec_R,
            Lp=Lp_beam,
            N_target=0.0,  # overwritten during assembly from beam axial force
            kappa_factor=-2.0,
            moment_sign=+1.0,
            tol_N=100.0,   
            max_iter_eps0=80,
            line_search=bool(fiber_line_search),
        )

        # FrameElementLinear2D reports N in tension-positive convention, while the
        # fiber section uses compression-positive. beam_sign=-1 converts conventions.
        hinges.append(
            FiberRotSpringElement(
                2,
                i2B,
                fiber_left,
                nodes,
                beam_idx=int(beam_elems[0]),
                beam_sign=-1.0,
                name="beam_hinge_L",
            )
        )
        hinges.append(
            FiberRotSpringElement(
                3,
                i3B,
                fiber_right,
                nodes,
                beam_idx=int(beam_elems[-1]),
                beam_sign=-1.0,
                name="beam_hinge_R",
            )
        )
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
    # Gravity load: distribute the total vertical gravity load across *all* frame
    # elements (columns + roof beam), using *physical* member areas (A_col/A_beam)
    # and physical member lengths (H/L).
    #
    # NOTE: In Problem 4 we use A_eff/I_eff to keep the elastic stiffness correct
    # after extracting plastic hinge zones. Those A_eff values should NOT be used to
    # compute gravity (self-weight) because they are not physical. Using A_col/A_beam
    # keeps Problem 4 and Problem 6 directly comparable.
    vol_total = 2.0 * float(A_col) * float(H) + float(A_beam) * float(L)
    w_gravity_col = 0.0
    w_gravity_beam = 0.0

    if vol_total > 0.0:
        w_gravity_col = float(P_gravity_total) * float(A_col) / vol_total  # N/m
        w_gravity_beam = float(P_gravity_total) * float(A_beam) / vol_total  # N/m

        col_eids = set(left_elems + right_elems)
        beam_eids = set(beam_elems)

        for eid, e in enumerate(beams):
            if eid in col_eids:
                w_e = w_gravity_col
            elif eid in beam_eids:
                w_e = w_gravity_beam
            else:
                # Fallback (should not happen): weight by element effective area.
                w_e = float(P_gravity_total) * float(e.A) / max(vol_total, 1e-12)
            f_g = e.equiv_nodal_load_global((0.0, -w_e))
            dofs = e.dofs()
            for a, ia in enumerate(dofs):
                p0[ia] += f_g[a]

    uy_dofs_unique = np.unique(np.array([n.dof_u[1] for n in nodes], dtype=int))
    gravity_Fy_total = float(np.sum(p0[uy_dofs_unique]))
    model = Model(
        nodes=nodes,
        beams=beams,
        hinges=hinges,
        fixed_dofs=fixed,
        mass_diag=mass,
        C_diag=C,
        load_const=p0,
        col_hinge_groups=[(idx_col_L, int(left_elems[0]), -1), (idx_col_R, int(right_elems[0]), -1)],
        nlgeom=nlgeom,
    )

    K_story = story_stiffness_linear(model, top_nodes=(2, 3))
    omega0 = 2.0 * math.pi / T0
    M_total = K_story / (omega0 ** 2)

    mass_mode = str(mass_mode).lower().strip()
    if mass_mode == "roof":
        # Classic SDOF-style roof lumping (horizontal only)
        mass[nodes[2].dof_u[0]] = 0.5 * M_total
        mass[nodes[3].dof_u[0]] = 0.5 * M_total
    elif mass_mode == "distributed":
        # Distribute the same total physical mass along all members and to all DOFs.
        distribute_lumped_mass_from_beams(
            nodes=nodes,
            beams=beams,
            mass=mass,
            M_total=M_total,
            include_rot_inertia=bool(include_rot_inertia),
        )
    else:
        raise ValueError(f"Unknown mass_mode: {mass_mode!r} (use 'roof' or 'distributed')")

    # Optional explicit helper: ensure strictly positive inertia/mass on all free DOFs.
    if explicit_mass_dt is not None:
        dt_target = float(explicit_mass_dt)
        u0 = np.zeros(nd)
        model.update_column_yields(u0)
        K0, _, _ = model.assemble(u0, u0)  # reduced to free DOFs
        fd = model.free_dofs()
        kdiag = np.maximum(np.diag(K0), 0.0)
        m_req = kdiag * (0.5 * dt_target) ** 2
        m_fd = mass[fd]
        m_fd = np.maximum(m_fd, m_req)
        m_fd = np.maximum(m_fd, 1e-12)  # strictly positive
        mass[fd] = m_fd


    C[:] = 2.0 * zeta * omega0 * mass

    # Mass bookkeeping (unique DOF indices; auxiliary nodes may share translations)
    ux_dofs = np.unique(np.array([n.dof_u[0] for n in nodes], dtype=int))
    uy_dofs = np.unique(np.array([n.dof_u[1] for n in nodes], dtype=int))
    th_dofs = np.unique(np.array([n.dof_th for n in nodes], dtype=int))
    mass_ux_total = float(np.sum(mass[ux_dofs]))
    mass_uy_total = float(np.sum(mass[uy_dofs]))
    inertia_th_total = float(np.sum(mass[th_dofs]))


    meta = {
        "K_story": K_story,
        "M_total": M_total,
        "mass_mode": str(mass_mode),
        "include_rot_inertia": bool(include_rot_inertia),
        "explicit_mass_dt": explicit_mass_dt,
        "mass_ux_total": mass_ux_total,
        "mass_uy_total": mass_uy_total,
        "inertia_th_total": inertia_th_total,
        "T0": T0,
        "omega0": omega0,
        "section_col": sec_col,
        "surface_col": surf_col,
        "section_beam": sec_beam,
        "surface_beam": surf_beam,
        "My_beam": My_beam,
        "Lp_col": Lp_col,
        "Lp_beam": Lp_beam,
        "Le_col": Le_col,
        "Le_beam": Le_beam,
        "s_col": s_col,
        "s_beam": s_beam,
        "A_col_eff": A_col_eff,
        "I_col_eff": I_col_eff,
        "A_beam_eff": A_beam_eff,
        "I_beam_eff": I_beam_eff,
        "H": H,
        "L": L,
        "nseg": nseg_use,
        "member_elements": {"left": left_elems, "right": right_elems, "beam": beam_elems},

        # Loading (gravity reference)
"P_gravity_total": float(P_gravity_total),
"gravity_scheme": "all_elements_area_length",
"gravity_vol_total": float(vol_total),
"gravity_w_col": float(w_gravity_col),
"gravity_w_beam": float(w_gravity_beam),
"gravity_Fy_total": float(gravity_Fy_total),
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
    max_cutbacks: int = 20,
    alpha=-0.05,
    nseg: int = 6,
    cover: float = 0.05,
    fiber_ny: int = 20,
    fiber_nz: int = 14,
    nlgeom: bool = False,
    beam_hinge: str = "shm",
    # Step 2 solver controls
    max_iter: int = 50,
    tol: float = 1e-6,
    # Step 1 gravity controls
    gravity_steps: int = 10,
    gravity_max_iter: int = 80,
    gravity_tol: float = 1e-10,
    gravity_verbose: bool = False,
    # Diagnostics
    debug_cutback: bool = False,
    # Optional: backtracking line search inside Newton (gravity + implicit dynamics)
    line_search: bool = False,
    out_dir: Path | None = None,
):
    """Incremental Dynamic Analysis with a reusable gravity preload.

    Workflow:
      Step 1: Gravity (static) -> committed hinge states + u_grav
      Step 2: Dynamic (IDA)    -> start each run from the gravity-preloaded state

    Notes:
      - Gravity is solved once per model build (per hinge option).
      - Each dt cutback attempt starts from a pristine deep-copy of the gravity-preloaded model.
    """

    import copy

    if float(base_dt) <= 0.0:
        raise ValueError("base_dt must be positive")
    if float(dt_min) <= 0.0:
        raise ValueError("dt_min must be positive")
    if int(max_cutbacks) < 0:
        raise ValueError("max_cutbacks must be non-negative")

    g = 9.81

    # Build model (contains hinges) and run gravity once.
    model0, meta = build_portal_beam_hinge(
        H=H,
        L=L,
        T0=T0,
        zeta=zeta,
        P_gravity_total=1500e3,
        nseg=nseg,
        cover=cover,
        nlgeom=nlgeom,
        beam_hinge=beam_hinge,
        fiber_ny=int(fiber_ny),
        fiber_nz=int(fiber_nz),
        fiber_line_search=bool(line_search),
    )

    grav = solve_gravity_only(
        model0,
        tol=float(gravity_tol),
        max_iter=int(gravity_max_iter),
        n_load_steps=int(gravity_steps),
        line_search=bool(line_search),
        verbose=bool(gravity_verbose),
    )
    u_grav = np.array(grav["u"], dtype=float)

    # Snapshot a reusable gravity-preloaded template.
    model_tpl = copy.deepcopy(model0)

    meta.update(
        {
            "line_search": bool(line_search),
            "H": float(H),
            "L": float(L),
            "T0": float(T0),
            "zeta": float(zeta),
            "gravity": {
                "steps": int(gravity_steps),
                "max_iter": int(gravity_max_iter),
                "tol": float(gravity_tol),
                "ux_roof": float(grav.get("ux_roof", float("nan"))),
                "uy_roof": float(grav.get("uy_roof", float("nan"))),
                "drift": float(grav.get("drift", float("nan"))),
            },
            "dynamic": {
                "base_dt": float(base_dt),
                "dt_min": float(dt_min),
                "max_cutbacks": int(max_cutbacks),
            },
        }
    )

    # --- Mass / excitation sanity checks (debug) ---
    nd = model_tpl.ndof()
    fd = model_tpl.free_dofs()

    # Unique translational DOFs (aux nodes may share translations)
    ux_dofs = np.unique(np.array([n.dof_u[0] for n in model_tpl.nodes], dtype=int))
    uy_dofs = np.unique(np.array([n.dof_u[1] for n in model_tpl.nodes], dtype=int))

    ux_free = np.intersect1d(ux_dofs, fd, assume_unique=False)
    uy_free = np.intersect1d(uy_dofs, fd, assume_unique=False)

    mass_ux_total = float(np.sum(model_tpl.mass_diag[ux_dofs]))
    mass_uy_total = float(np.sum(model_tpl.mass_diag[uy_dofs]))
    mass_ux_free_total = float(np.sum(model_tpl.mass_diag[ux_free]))
    mass_uy_free_total = float(np.sum(model_tpl.mass_diag[uy_free]))

    # Influence vector for horizontal base excitation (ux only)
    r = np.zeros(nd)
    for node in model_tpl.nodes:
        r[node.dof_u[0]] = 1.0
    m_eff_ux_free = float(np.sum((model_tpl.mass_diag * r)[fd]))

    meta.update(
        {
            "mass_ux_total": mass_ux_total,
            "mass_uy_total": mass_uy_total,
            "mass_ux_free_total": mass_ux_free_total,
            "mass_uy_free_total": mass_uy_free_total,
            "m_eff_ux_free": m_eff_ux_free,
            "n_dof": int(nd),
            "n_free": int(fd.size),
            "n_mass_pos": int(np.sum(model_tpl.mass_diag > 0.0)),
        }
    )

    print("[problema4] mass sanity:")
    print(f"  ndof={nd} nfree={fd.size} n_mass_pos={meta['n_mass_pos']}")
    print(f"  M_ux_total={mass_ux_total:.6e}  M_ux_free={mass_ux_free_total:.6e}  m_eff_ux_free={m_eff_ux_free:.6e}")
    print(f"  M_uy_total={mass_uy_total:.6e}  M_uy_free={mass_uy_free_total:.6e}")

    if mass_ux_free_total <= 0.0 or not np.isfinite(mass_ux_free_total):
        raise RuntimeError(
            "Problema4: ux mass on FREE DOFs is zero/invalid -> base excitation produces ~0 inertial forcing. Check mass assignment."
        )

    peak_drifts: list[float] = []
    dt_hist: list[float] = []
    last = None
    model_last = model_tpl

    amps_g = np.array(list(amps_g), dtype=float)
    print(f"[problema4] Incremental dynamic analysis: {len(amps_g)} amplitudes, integrator={integrator}")
    for idx, A_g in enumerate(amps_g):
        A = float(A_g) * g
        dt = float(base_dt)
        cutbacks = 0

        while True:
            # Fresh copy per attempt (avoids polluting the committed hinge history on failed dt tries)
            model = copy.deepcopy(model_tpl)
            u0 = u_grav.copy()
            v0 = np.zeros(model.ndof())

            t = make_time(float(t_end), float(dt))
            ag = ag_fun(t, A)

            # effective inertial forcing amplitude on free DOFs (ux only)
            if debug_cutback:
                try:
                    nd_ = model.ndof()
                    fd_ = model.free_dofs()
                    r_ = np.zeros(nd_)
                    for node in model.nodes:
                        r_[node.dof_u[0]] = 1.0
                    f_in = (model.mass_diag * r_) * ag  # N = kg*m/s^2
                    fpk = float(np.max(np.abs(f_in[fd_]))) if fd_.size else float("nan")
                    print(f"    dt={dt:.6f}  ag_pk={float(np.max(np.abs(ag))):.4f}  |F_inert|_pk_free={fpk:.6e}")
                except Exception:
                    pass

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
                    max_iter=int(max_iter),
                    tol=float(tol),
                    # Explicit can look "zombie" (no Newton prints). Provide a small
                    # header + stability substepping info by default.
                    verbose=bool(debug_cutback) or (integrator == "explicit"),
                    line_search=bool(line_search),
                    u0=u0,
                    v0=v0,
                )
                out["A_ms2"] = float(A)
                out["A_input_g"] = float(A_g)
                out["zeta"] = float(zeta)
                out["T0"] = float(T0)

                # Export per-anregung diagnostics (energy balance + excitation)
                if out_dir is not None:
                    try:
                        ida_dir = Path(out_dir) / "ida"
                        ida_dir.mkdir(parents=True, exist_ok=True)
                        prefix = f"problem4_Ag_{A_g:.2f}g"
                        export_anregung(ida_dir, prefix, out["t"], out["ag"])
                        if isinstance(out.get("energy", None), dict):
                            export_energy_balance(ida_dir, prefix, out["t"], out["energy"])
                        # Compact NPZ for re-plotting
                        e = out.get("energy", {}) if isinstance(out.get("energy", None), dict) else {}
                        np.savez_compressed(
                            ida_dir / f"{prefix}_history.npz",
                            t=out["t"], ag=out["ag"], drift=out.get("drift", None), Vb=out.get("Vb", None),
                            T=e.get("T", None), W_ext=e.get("W_ext", None), W_int=e.get("W_int", None),
                            W_damp=e.get("W_damp", None), W_pl=e.get("W_pl", None), residual=e.get("residual", None),
                        )
                    except Exception:
                        pass

                last = out
                model_last = model
                dt_hist.append(float(out["dt"]))
                break
            except RuntimeError as e:
                if debug_cutback:
                    print(f"    cutback: A_g={A_g:.2f} dt={dt:g} failed: {e}")
                if dt <= float(dt_min) + 1e-15 or cutbacks >= int(max_cutbacks):
                    reason = "dt<dtmin" if dt <= float(dt_min) + 1e-15 else f"max_cutbacks={int(max_cutbacks)}"
                    print(f"  [{idx+1}/{len(amps_g)}] A_g={A_g:.2f} - COLLAPSE ({reason})")
                    meta.update({"dt_hist": dt_hist, "amps_g_used": amps_g[:len(peak_drifts)]})
                    return peak_drifts, amps_g[:len(peak_drifts)], last, model_last, meta
                dt *= 0.5
                cutbacks += 1
                continue

        pk = float(np.max(np.abs(last["drift"]))) if last is not None else float("nan")
        peak_drifts.append(pk)
        pct = 100.0 * (idx + 1) / len(amps_g)
        print(
            f"  [{idx+1}/{len(amps_g)}] ({pct:5.1f}%) A_g={A_g:.2f} drift_max={pk:.4f} dt={last['dt']:.5f} n_steps={len(last['t'])}"
        )
        if pk >= float(drift_limit):
            print(f"  DRIFT LIMIT REACHED: {pk:.4f} >= {float(drift_limit):.4f}")
            break

    meta.update({"dt_hist": dt_hist, "amps_g_used": amps_g[:len(peak_drifts)]})
    return peak_drifts, amps_g[:len(peak_drifts)], last, model_last, meta


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

    # Convenience exports for the *last* run in the root output folder
    # (IDA per-amplitude exports go into out_dir/ida when out_dir is provided).
    try:
        Ag = last.get("A_input_g", None)
        prefix = f"problem4_last_Ag_{float(Ag):.2f}g" if Ag is not None else "problem4_last"
        export_anregung(out, prefix, last["t"], last["ag"])
        if isinstance(last.get("energy", None), dict):
            export_energy_balance(out, prefix, last["t"], last["energy"])
        e = last.get("energy", {}) if isinstance(last.get("energy", None), dict) else {}
        np.savez_compressed(
            out / f"{prefix}_history.npz",
            t=last.get("t", None),
            ag=last.get("ag", None),
            drift=last.get("drift", None),
            Vb=last.get("Vb", None),
            T=e.get("T", None),
            W_ext=e.get("W_ext", None),
            W_int=e.get("W_int", None),
            W_damp=e.get("W_damp", None),
            W_pl=e.get("W_pl", None),
            residual=e.get("residual", None),
        )
    except Exception:
        pass




# -------------------------------------------------------------------------
# Optional reference: build + solve Problema 6 (elastic) for gravity comparison
# This helper is intentionally defensive: if Problema 6 is unavailable, it
# returns None and the caller should skip the comparison.
def _try_build_problem6_reference(nseg: int, nlgeom: bool):
    try:
        import importlib
        import inspect
        from dc_solver.utils.gravity import solve_gravity_only

        mod = importlib.import_module("problems.problema6_portico_elastico")

        # Find a reasonable "builder" function in Problema 6
        builder = None
        for name in (
            "build_problem6_model",
            "build_portal_elastic",
            "build_portal_elastic_model",
            "build_portal_model",
            "build_model",
        ):
            if hasattr(mod, name):
                builder = getattr(mod, name)
                break

        if builder is None:
            # Fallback: if the module exposes a 'make_model' helper
            if hasattr(mod, "make_model"):
                builder = getattr(mod, "make_model")
            else:
                return None

        sig = inspect.signature(builder)
        kwargs = {}
        if "nseg" in sig.parameters:
            kwargs["nseg"] = int(nseg)
        elif "n_seg" in sig.parameters:
            kwargs["n_seg"] = int(nseg)

        if "nlgeom" in sig.parameters:
            kwargs["nlgeom"] = bool(nlgeom)
        elif "nonlinear_geometry" in sig.parameters:
            kwargs["nonlinear_geometry"] = bool(nlgeom)

        out = builder(**kwargs)
        if isinstance(out, tuple) and len(out) >= 1:
            model6 = out[0]
            meta6 = out[1] if len(out) > 1 else {}
        else:
            model6 = out
            meta6 = {}

        res6 = solve_gravity_only(model6)
        if isinstance(res6, dict):
            res6["model"] = model6
        res6["meta"] = meta6
        return res6
    except Exception:
        return None
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Problema 4: pórtico con rótulas (Step 1: gravity, Step 2: dynamic/IDA).")
    # Mode selector
    parser.add_argument("--state", default="ida", choices=["ida", "gravity"],
                        help="Run mode: 'gravity' runs Step 1 only; 'ida' runs Step 1 then Step 2 (incremental dynamic).")
    parser.add_argument("--gravity", action="store_true",
                        help="Alias for --state gravity (run only Step 1).")

    # Model/integration
    parser.add_argument("--integrator", default="hht", choices=["hht", "newmark", "explicit"], help="Time integrator.")
    parser.add_argument("--nlgeom", action="store_true", help="Enable geometric nonlinearity (P-Delta).")
    parser.add_argument("--nseg", type=int, default=6, help="Segments per member (visualization/mesh).")
    parser.add_argument(
        "--beam-hinge",
        default="shm",
        choices=["shm", "fiber", "compare"],
        help="Beam end hinge model: 'shm', 'fiber', or 'compare' (runs both).",
    )

    # Step 1 (gravity) controls
    parser.add_argument("--gravity-steps", type=int, default=10, help="Load steps for gravity ramp (Step 1).")
    parser.add_argument("--gravity-max-iter", type=int, default=80, help="Max Newton iterations per gravity load step.")
    parser.add_argument("--gravity-tol", type=float, default=1e-10, help="Gravity Newton tolerance (relative).")
    parser.add_argument("--gravity-verbose", action="store_true", help="Verbose output for gravity Newton.")

    # Step 2 (dynamic / IDA) controls
    parser.add_argument("--t-end", type=float, default=10.0, help="End time for dynamic step [s].")
    parser.add_argument("--base-dt", type=float, default=0.002, help="Initial time step for dynamic step [s].")
    parser.add_argument("--dt-min", type=float, default=0.00025, help="Minimum allowed cutback dt [s].")
    parser.add_argument("--max-cutbacks", type=int, default=20, help="Maximum dt cutbacks per IDA amplitude.")
    parser.add_argument("--max-iter", type=int, default=50, help="Max Newton iterations per time step (implicit integrators).")
    parser.add_argument("--tol", type=float, default=1e-6, help="Newton tolerance for dynamic step.")
    parser.add_argument("--alpha", type=float, default=-0.05, help="HHT-alpha parameter (only for HHT).")
    parser.add_argument("--drift-limit", type=float, default=0.10, help="Collapse drift limit.")
    parser.add_argument("--snapshot-limit", type=float, default=0.04, help="Drift at which to snapshot plots.")

    # IDA amplitudes (in g)
    parser.add_argument("--ag-min", type=float, default=0.10, help="Minimum A_g to run [g].")
    parser.add_argument("--ag-max", type=float, default=2.00, help="Maximum A_g to run [g].")
    parser.add_argument("--ag-step", type=float, default=0.10, help="A_g increment [g].")

    # Record controls
    parser.add_argument("--T0", type=float, default=0.5, help="Target fundamental period for synthetic record [s].")
    parser.add_argument("--zeta", type=float, default=0.02, help="Modal damping ratio for synthetic record.")

    # Debugging
    parser.add_argument("--debug-cutback", action="store_true", help="Print details for dt cutbacks (failed attempts).")

    # Newton robustness
    parser.add_argument("--line-search", action="store_true", help="Enable backtracking line search in Newton solvers (gravity + dynamic + fiber eps0 solve).")

    # Fiber-section discretization (only used for --beam-hinge fiber/compare)
    parser.add_argument("--fiber-ny", type=int, default=20, help="Fiber mesh divisions in depth (y) for RC section.")
    parser.add_argument("--fiber-nz", type=int, default=14, help="Fiber mesh divisions in width (z) for RC section.")

    args = parser.parse_args()

    # convenience alias
    if getattr(args, "gravity", False):
        args.state = "gravity"


    # --- State selector ---------------------------------------------------------
    if args.state == "gravity":
        out_dir = _outputs_dir(f"problem4_{args.beam_hinge}_gravity")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build Problem 4 model (with hinges) and solve static gravity equilibrium.
        # NOTE: this repo already provides `build_portal_beam_hinge(...)` as the
        # canonical builder for Problema 4.
        model4, _meta4 = build_portal_beam_hinge(
            nseg=int(args.nseg),
            nlgeom=bool(args.nlgeom),
            beam_hinge=args.beam_hinge,
            # Use distributed mass by default for consistency with dynamic runs
            mass_mode="distributed",
            fiber_ny=int(getattr(args, 'fiber_ny', 20)),
            fiber_nz=int(getattr(args, 'fiber_nz', 14)),
            fiber_line_search=bool(getattr(args, 'line_search', False)),
        )
        r4 = solve_gravity_only(
            model4,
            tol=float(args.gravity_tol),
            max_iter=int(args.gravity_max_iter),
            n_load_steps=int(args.gravity_steps),
            verbose=bool(args.gravity_verbose),
            line_search=bool(getattr(args, 'line_search', False)),
        )

        meta4 = dict(_meta4) if isinstance(_meta4, dict) else {}

        # Try to build Problem 6 elastic reference (if available in this repo)
        r6 = _try_build_problem6_reference(nseg=int(args.nseg), nlgeom=bool(args.nlgeom))
        meta6 = dict(r6.get("meta", {})) if isinstance(r6, dict) else {}

        txt = []
        # --- Load verification -------------------------------------------------
        try:
            P4 = float(meta4.get("P_gravity_total", float("nan")))
            Fy4 = float(meta4.get("gravity_Fy_total", float("nan")))
            V4 = float(meta4.get("gravity_vol_total", float("nan")))
            w4c = float(meta4.get("gravity_w_col", float("nan")))
            w4b = float(meta4.get("gravity_w_beam", float("nan")))

            P6 = float(meta6.get("P_gravity_total", float("nan")))
            Fy6 = float(meta6.get("gravity_Fy_total", float("nan")))
            V6 = float(meta6.get("gravity_vol_total", float("nan")))
            w6c = float(meta6.get("gravity_w_col", float("nan")))
            w6b = float(meta6.get("gravity_w_beam", float("nan")))

            txt.append("Gravity load check (distributed self-weight on all frame elements)\n")
            txt.append(
                f"  P4: P={P4:.6e}N, vol=Σ(A·L)={V4:.6e} m^3, w_col={w4c:.6e}N/m, w_beam={w4b:.6e}N/m, "
                f"sum(Fy)={Fy4:.6e}N, sum(Fy)+P={Fy4+P4:.3e}\n"
            )
            if not math.isnan(P6):
                txt.append(
                    f"  P6: P={P6:.6e}N, vol=Σ(A·L)={V6:.6e} m^3, w_col={w6c:.6e}N/m, w_beam={w6b:.6e}N/m, "
                    f"sum(Fy)={Fy6:.6e}N, sum(Fy)+P={Fy6+P6:.3e}\n"
                )
            txt.append("\n")
        except Exception:
            pass

        # --- SHM hinge parameter check (elastic-range stiffness) --------------
        if str(args.beam_hinge).lower() == "shm":
            try:
                from dc_solver.hinges.models import RotSpringElement as _RotSpring

                txt.append("SHM hinge check (elastic-range stiffness)\n")
                for h in getattr(model4, "hinges", []):
                    if isinstance(h, _RotSpring) and h.kind == "beam_shm" and h.beam_hinge is not None:
                        K0 = float(h.beam_hinge.K0_0)
                        My = float(h.beam_hinge.My_0)
                        A_eff = float(h.beam_hinge.bw_A) if float(h.beam_hinge.bw_A) > 0.0 else float(K0 / max(abs(My), 1e-12))
                        txt.append(f"  beam_shm: K0_0={K0:.6e} My_0={My:.6e} bw_A_eff~{A_eff:.6e} (target bw_A_eff≈K0/My)\n")
                txt.append("\n")
            except Exception:
                pass

        txt.append("Problem 4 gravity-only (with hinges)\n")
        ux4 = float(r4.get("ux_roof", float("nan")))
        uy4 = float(r4.get("uy_roof", float("nan")))
        dr4 = float(r4.get("drift", float("nan")))
        vb4 = float(r4.get("Vb", float("nan")))
        txt.append(f"ux_roof={ux4:.6e} uy_roof={uy4:.6e} drift={dr4:.6e}\n")
        txt.append(f"Vb={vb4:.6e}\n\n")

        if r6 is not None:
            txt.append("Problem 6 gravity-only (elastic reference)\n")
            ux6 = float(r6.get("ux_roof", float("nan")))
            uy6 = float(r6.get("uy_roof", float("nan")))
            dr6 = float(r6.get("drift", float("nan")))
            vb6 = float(r6.get("Vb", float("nan")))
            txt.append(f"ux_roof={ux6:.6e} uy_roof={uy6:.6e} drift={dr6:.6e}\n")
            txt.append(f"Vb={vb6:.6e}\n\n")
            txt.append("Delta (P4 - P6)\n")
            txt.append(f"dUx={(ux4-ux6):.6e} dUy={(uy4-uy6):.6e}\n")
        else:
            txt.append("Problem 6 reference not available (import failed).\n")

        out_dir.joinpath("gravity_compare.txt").write_text("".join(txt), encoding="utf-8")
        print("".join(txt))

        # Optional: dump fiber beam hinge axial targets (for debugging N_target coupling)
        try:
            u_g = r4.get("u", None)
            if isinstance(u_g, np.ndarray) and args.beam_hinge in ("fiber", "compare"):
                _, _, inf = model4.assemble(u_g, u_g)
                rows = []
                for hinfo in (inf.get("hinges") or []):
                    if str(hinfo.get("kind", "")) == "beam_fiber":
                        rows.append(
                            f"{hinfo.get('name','(beam_fiber)')}: "
                            f"beam_idx={hinfo.get('beam_idx')} "
                            f"N_beam_tension={float(hinfo.get('N_beam_tension', 0.0)):.6e} "
                            f"N_target_used={float(hinfo.get('N_target_used', hinfo.get('N_target', 0.0))):.6e} "
                            f"N_section={float(hinfo.get('N', 0.0)):.6e}"
                        )
                if rows:
                    out_dir.joinpath("fiber_hinge_axial_debug.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
                    if bool(args.gravity_verbose):
                        print("[gravity] Fiber hinge axial targets (compression-positive in section):")
                        for r in rows:
                            print("  " + r)
        except Exception:
            pass

        print(f"[problema4] Gravity-only outputs in: {out_dir.resolve()}")

        # --- Gravity plots (State 1: undeformed, State 2: static equilibrium) ---
        try:
            u_g = r4.get("u", None)
            if isinstance(u_g, np.ndarray):
                # Build a minimal 'last' dict compatible with plot_structure_states
                u_hist = np.vstack([np.zeros_like(u_g), u_g])
                drift_hist = np.array([0.0, float(r4.get("drift", 0.0))], float)
                last_grav = {
                    "u": u_g[np.newaxis, :],      # shape (1, ndof)
                    "t": np.array([0.0], float),
                    "drift": np.array([float(r4.get("drift", 0.0))], float),
                    "snapshot_limit": float(args.snapshot_limit),
                }       

                # Use geometric height from the model (avoid relying on hard-coded H)
                ys = np.array([nd.y for nd in model4.nodes], dtype=float)
                H_plot = float(np.max(ys) - np.min(ys)) if ys.size else 1.0
                if not np.isfinite(H_plot) or H_plot <= 0.0:
                    H_plot = 1.0

                # combined (members), and individual fields
                plot_structure_states(
                    model4,
                    last_grav,
                    drift_height=H_plot,
                    snapshot_limit=float(args.snapshot_limit),
                    outfile=str(out_dir / "step1_gravity_states_members.png"),
                    field="both",
                    shared_colorbar=False,
                )
                plot_structure_states(
                    model4,
                    last_grav,
                    drift_height=H_plot,
                    snapshot_limit=float(args.snapshot_limit),
                    outfile=str(out_dir / "step1_gravity_states_U.png"),
                    field="u",
                    shared_colorbar=False,
                )
                plot_structure_states(
                    model4,
                    last_grav,
                    drift_height=H_plot,
                    snapshot_limit=float(args.snapshot_limit),
                    outfile=str(out_dir / "step1_gravity_states_S.png"),
                    field="s",
                    shared_colorbar=False,
                )

                # CSV for member stress summary at gravity state
                write_member_stress_csv(model4, u_g, out_dir / "step1_gravity_member_stress.csv")
        except Exception as e:
            # Plotting must never fail the run
            (out_dir / "step1_gravity_plot_error.txt").write_text(str(e), encoding="utf-8")
        except Exception as e:
            # Plotting must never fail the run
            (out_dir / "step1_gravity_plot_error.txt").write_text(str(e), encoding="utf-8")
        return



    drift_limit = float(args.drift_limit)
    snapshot_limit = float(args.snapshot_limit)
    alpha = float(args.alpha)
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
            "line_search": bool(meta.get("line_search", False)),
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
            T0=float(args.T0),
            zeta=float(args.zeta),
            drift_limit=drift_limit,
            amps_g=np.arange(float(args.ag_min), float(args.ag_max) + 0.5*float(args.ag_step), float(args.ag_step)),
            t_end=float(args.t_end),
            base_dt=float(args.base_dt),
            dt_min=float(args.dt_min),
            max_cutbacks=int(args.max_cutbacks),
            alpha=alpha,
            nseg=nseg,
            nlgeom=nlgeom,
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            gravity_steps=int(args.gravity_steps),
            gravity_max_iter=int(args.gravity_max_iter),
            gravity_tol=float(args.gravity_tol),
            gravity_verbose=bool(args.gravity_verbose),
            debug_cutback=bool(args.debug_cutback),
            line_search=bool(args.line_search),
            beam_hinge=args.beam_hinge,
            fiber_ny=int(getattr(args, "fiber_ny", 20)),
            fiber_nz=int(getattr(args, "fiber_nz", 14)),
            out_dir=out,
        )
        plot_results(last, model, meta, drift_limit=drift_limit, snapshot_limit=snapshot_limit, outdir=out)
        _write_basic_plots(out, last)
        _write_summary(out, meta, amps_used, peak_drifts)
        _write_runinfo(out, meta, last)
        print(f"\n[problema4] Analysis complete. Outputs in: {out}")
        return

    # compare mode
    out_shm = _outputs_dir(f"problem4_shm_{args.integrator}")
    out_fib = _outputs_dir(f"problem4_fiber_{args.integrator}")
    out_cmp = _outputs_dir(f"problem4_compare_{args.integrator}")

    pd_shm, amps_shm, last_shm, model_shm, meta_shm = run_incremental_amplitudes(
        integrator=args.integrator,
        H=3.0,
        L=5.0,
        T0=float(args.T0),
        zeta=float(args.zeta),
        drift_limit=drift_limit,
        amps_g=np.arange(float(args.ag_min), float(args.ag_max) + 0.5*float(args.ag_step), float(args.ag_step)),
        t_end=float(args.t_end),
        base_dt=float(args.base_dt),
        dt_min=float(args.dt_min),
        max_cutbacks=int(args.max_cutbacks),
        alpha=alpha,
        nseg=nseg,
        nlgeom=nlgeom,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        gravity_steps=int(args.gravity_steps),
        gravity_max_iter=int(args.gravity_max_iter),
        gravity_tol=float(args.gravity_tol),
        gravity_verbose=bool(args.gravity_verbose),
        debug_cutback=bool(args.debug_cutback),
        beam_hinge="shm",
        line_search=bool(getattr(args, "line_search", False)),
        fiber_ny=int(getattr(args, "fiber_ny", 20)),
        fiber_nz=int(getattr(args, "fiber_nz", 14)),
        out_dir=out_shm,
    )
    plot_results(last_shm, model_shm, meta_shm, drift_limit=drift_limit, snapshot_limit=snapshot_limit, outdir=out_shm)
    _write_basic_plots(out_shm, last_shm)
    _write_summary(out_shm, meta_shm, amps_shm, pd_shm)
    _write_runinfo(out_shm, meta_shm, last_shm)

    pd_fib, amps_fib, last_fib, model_fib, meta_fib = run_incremental_amplitudes(
        integrator=args.integrator,
        H=3.0,
        L=5.0,
        T0=float(args.T0),
        zeta=float(args.zeta),
        drift_limit=drift_limit,
        amps_g=np.arange(float(args.ag_min), float(args.ag_max) + 0.5*float(args.ag_step), float(args.ag_step)),
        t_end=float(args.t_end),
        base_dt=float(args.base_dt),
        dt_min=float(args.dt_min),
        max_cutbacks=int(args.max_cutbacks),
        alpha=alpha,
        nseg=nseg,
        nlgeom=nlgeom,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        gravity_steps=int(args.gravity_steps),
        gravity_max_iter=int(args.gravity_max_iter),
        gravity_tol=float(args.gravity_tol),
        gravity_verbose=bool(args.gravity_verbose),
        debug_cutback=bool(args.debug_cutback),
        beam_hinge="fiber",
        line_search=bool(getattr(args, "line_search", False)),
        fiber_ny=int(getattr(args, "fiber_ny", 20)),
        fiber_nz=int(getattr(args, "fiber_nz", 14)),
        out_dir=out_fib,
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

    print(f"\n[problema4] Comparison analysis complete.")
    print(f"  SHM outputs:     {out_shm}")
    print(f"  Fiber outputs:   {out_fib}")
    print(f"  Compare outputs: {out_cmp}")


if __name__ == "__main__":
    main()
