"""Portal frame elastic test (Problema 6).

Purpose
-------
A clean regression / sanity test for the dynamic integrators (Newmark / HHT / Explicit)
using the same portal geometry as Problema 4 but **without plastic hinges**.

Key points:
- Drift is reported as (roof_avg_ux - base_avg_ux) / H.
- Ground acceleration enters as -M r ag (relative coordinates).
- Mass is distributed along all members and assigned to **all DOFs** (ux, uy, theta).

Added:
- --state gravity : run a gravity-only static Newton solve (State 1) to compare against Problema 4.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dc_solver.fem.nodes import Node, DofManager
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model
from dc_solver.fem.utils import discretize_member
from dc_solver.integrators import solve_dynamic
from dc_solver.reporting.run_info import build_run_info, write_run_info
from dc_solver.post.plotting import plot_structure_states

# NEW: gravity-only helper (returns u + ux_roof/uy_roof/drift/Vb)
from dc_solver.utils.gravity import solve_gravity_only


def _outputs_dir(subdir: Optional[str] = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    if subdir:
        out = out / str(subdir)
        out.mkdir(parents=True, exist_ok=True)
    return out


def make_time(t_end: float, dt: float) -> np.ndarray:
    return np.arange(0.0, t_end + 1e-12, dt)


def ag_fun(t: np.ndarray, A: float) -> np.ndarray:
    # same excitation as problem 4
    return A * np.cos(0.2 * np.pi * t) * np.sin(4.0 * np.pi * t)


def distribute_lumped_mass_from_beams(
    *,
    nodes: List[Node],
    beams: List[FrameElementLinear2D],
    mass: np.ndarray,
    M_total: float,
    include_rot_inertia: bool = True,
) -> None:
    """Distribute a target total physical mass along all frame members (lumped, diagonal)."""
    total_L = sum(float(e._geom()[0]) for e in beams)
    if total_L <= 0.0 or M_total <= 0.0:
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


def build_portal_elastic(
    *,
    H: float = 3.0,
    L: float = 5.0,
    T0: float = 0.5,
    zeta: float = 0.02,
    P_gravity_total: float = 1500e3,
    nseg: int = 6,
    nlgeom: bool = False,
    mass_mode: str = "distributed",  # "roof" | "distributed"
    include_rot_inertia: bool = True,
    explicit_mass_dt: Optional[float] = None,
) -> Tuple[Model, Dict]:
    dm = DofManager()

    n0 = Node(0.0, 0.0, dm.new_trans(), dm.new_rot())
    n1 = Node(L, 0.0, dm.new_trans(), dm.new_rot())
    n2 = Node(0.0, H, dm.new_trans(), dm.new_rot())
    n3 = Node(L, H, dm.new_trans(), dm.new_rot())

    nodes: List[Node] = [n0, n1, n2, n3]

    # Discretize members (no aux nodes / no hinges)
    left_nodes = discretize_member(0, 2, int(nseg), nodes, dm)
    right_nodes = discretize_member(1, 3, int(nseg), nodes, dm)
    beam_nodes = discretize_member(2, 3, int(nseg), nodes, dm)

    E = 30e9
    b_col, h_col = 0.40, 0.60
    b_beam, h_beam = 0.25, 0.50
    A_col = b_col * h_col
    I_col = b_col * (h_col ** 3) / 12.0
    A_beam = b_beam * h_beam
    I_beam = b_beam * (h_beam ** 3) / 12.0

    beams: List[FrameElementLinear2D] = []

    def add_member(node_ids: List[int], A: float, I: float) -> None:
        for a, b in zip(node_ids[:-1], node_ids[1:]):
            beams.append(FrameElementLinear2D(a, b, E=E, A=A, I=I, nodes=nodes))

    add_member(left_nodes, A_col, I_col)
    add_member(right_nodes, A_col, I_col)
    add_member(beam_nodes, A_beam, I_beam)

    fixed = np.array([
        nodes[0].dof_u[0], nodes[0].dof_u[1], nodes[0].dof_th,
        nodes[1].dof_u[0], nodes[1].dof_u[1], nodes[1].dof_th,
    ], dtype=int)

    nd = dm.ndof
    mass = np.zeros(nd)
    C = np.zeros(nd)
    p0 = np.zeros(nd)
    # Gravity load: distribute the total vertical gravity load across *all* frame
    # elements (columns + roof beam), proportional to element "volume" A*L.
    vol_total = 0.0
    for e in beams:
        L_e = float(e._geom()[0])
        vol_total += float(e.A) * L_e

    w_gravity_col = 0.0
    w_gravity_beam = 0.0

    if vol_total > 0.0:
        w_gravity_col = float(P_gravity_total) * float(A_col) / vol_total
        w_gravity_beam = float(P_gravity_total) * float(A_beam) / vol_total

        for e in beams:
            w_e = float(P_gravity_total) * float(e.A) / vol_total  # N/m
            f_g = e.equiv_nodal_load_global((0.0, -w_e))
            dofs = e.dofs()
            for a, ia in enumerate(dofs):
                p0[ia] += f_g[a]

    uy_dofs_unique = np.unique(np.array([n.dof_u[1] for n in nodes], dtype=int))
    gravity_Fy_total = float(np.sum(p0[uy_dofs_unique]))
    model = Model(
        nodes=nodes,
        beams=beams,
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=mass,
        C_diag=C,
        load_const=p0,
        col_hinge_groups=[],
        nlgeom=nlgeom,
    )

    # target period -> target total mass
    K_story = story_stiffness_linear(model, top_nodes=(2, 3))
    omega0 = 2.0 * math.pi / float(T0)
    M_total = K_story / (omega0 ** 2)

    mm = str(mass_mode).lower().strip()
    if mm == "roof":
        mass[nodes[2].dof_u[0]] = 0.5 * M_total
        mass[nodes[3].dof_u[0]] = 0.5 * M_total
    elif mm == "distributed":
        distribute_lumped_mass_from_beams(
            nodes=nodes,
            beams=beams,
            mass=mass,
            M_total=M_total,
            include_rot_inertia=bool(include_rot_inertia),
        )
    else:
        raise ValueError(f"Unknown mass_mode: {mass_mode!r} (use 'roof' or 'distributed')")

    # explicit stability helper (optional)
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
        m_fd = np.maximum(m_fd, 1e-12)
        mass[fd] = m_fd

    C[:] = 2.0 * float(zeta) * omega0 * mass

    # bookkeeping
    ux_dofs = np.unique(np.array([n.dof_u[0] for n in nodes], dtype=int))
    uy_dofs = np.unique(np.array([n.dof_u[1] for n in nodes], dtype=int))
    th_dofs = np.unique(np.array([n.dof_th for n in nodes], dtype=int))

    meta = {
        "H": float(H),
        "L": float(L),
        "nseg": int(nseg),
        "nlgeom": bool(nlgeom),
        "T0": float(T0),
        "zeta": float(zeta),
        "K_story": float(K_story),
        "omega0": float(omega0),
        "M_total": float(M_total),
        "mass_mode": str(mm),
        "include_rot_inertia": bool(include_rot_inertia),
        "mass_ux_total": float(np.sum(mass[ux_dofs])),
        "mass_uy_total": float(np.sum(mass[uy_dofs])),
        "inertia_th_total": float(np.sum(mass[th_dofs])),
        "explicit_mass_dt": float(explicit_mass_dt) if explicit_mass_dt is not None else None,

        # Loading (gravity reference)
"P_gravity_total": float(P_gravity_total),
"gravity_scheme": "all_elements_area_length",
"gravity_vol_total": float(vol_total),
"gravity_w_col": float(w_gravity_col),
"gravity_w_beam": float(w_gravity_beam),
"gravity_Fy_total": float(gravity_Fy_total),
    }
    return model, meta


def run_one(
    *,
    integrator: str,
    model: Model,
    meta: Dict,
    A_g: float,
    t_end: float,
    dt: float,
    drift_limit: float,
    snapshot_limit: float,
    alpha: float,
) -> Dict:
    g = 9.81
    t = make_time(float(t_end), float(dt))
    ag = ag_fun(t, float(A_g) * g)

    out = solve_dynamic(
        integrator,
        model=model,
        t=t,
        ag=ag,
        drift_height=float(meta["H"]),
        drift_limit=float(drift_limit),
        drift_snapshot=float(snapshot_limit),
        alpha=float(alpha),
        beta=0.25,
        gamma=0.50,
        base_nodes=(0, 1),
        drift_nodes=(2, 3),
        max_iter=50,
        tol=1e-8,
        verbose=False,
    )
    out["A_input_g"] = float(A_g)
    out["A_ms2"] = float(A_g) * g
    return out


def write_basic_plots(out: Path, last: Dict, drift_limit: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(last["t"], last["drift"], label="drift")
    ax.axhline(drift_limit, color="r", linestyle="--", label="limit")
    ax.axhline(-drift_limit, color="r", linestyle="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Drift ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem6_drift_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(last["drift"], last["Vb"])
    ax.set_xlabel("Drift ratio")
    ax.set_ylabel("Base shear [N]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "problem6_vb_drift.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Problema 6: pórtico ELÁSTICO (sin rótulas) para test integradores."
    )
    parser.add_argument(
        "--state",
        default="dynamic",
        choices=["dynamic", "gravity"],
        help="dynamic = time history, gravity = static Step 1 only.",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Alias for --state gravity (run only gravity step).",
    )

    parser.add_argument(
        "--integrator",
        default="newmark",
        choices=["hht", "newmark", "explicit"],
        help="Time integrator.",
    )
    parser.add_argument("--nlgeom", action="store_true", help="Enable geometric nonlinearity (P-Delta).")
    parser.add_argument("--nseg", type=int, default=6, help="Segments per member.")
    parser.add_argument(
        "--mass-mode",
        default="distributed",
        choices=["roof", "distributed"],
        help="Mass lumping strategy.",
    )
    parser.add_argument("--no-rot-inertia", action="store_true", help="Disable rotational inertia lumping.")
    parser.add_argument("--A_g", type=float, default=0.10, help="Peak amplitude in g (single run).")

    # Dynamic controls
    parser.add_argument("--t_end", type=float, default=10.0, help="End time [s].")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step [s].")
    # Aliases used in Problema 4
    parser.add_argument("--t-end", dest="t_end", type=float, help="Alias for --t_end [s].")
    parser.add_argument("--base-dt", dest="dt", type=float, help="Alias for --dt [s].")

    parser.add_argument("--alpha", type=float, default=-0.05, help="HHT-alpha parameter.")
    parser.add_argument("--drift-limit", type=float, default=0.10, help="Stop when max drift exceeds this limit.")
    parser.add_argument("--snapshot-limit", type=float, default=0.04, help="Record a snapshot at this drift limit.")

    # Gravity controls
    parser.add_argument("--gravity-steps", type=int, default=10, help="Load steps for gravity ramp (Step 1).")
    parser.add_argument("--gravity-max-iter", type=int, default=80, help="Max Newton iterations per load step (gravity).")
    parser.add_argument("--gravity-tol", type=float, default=1e-10, help="Gravity solver tolerance.")
    parser.add_argument("--gravity-verbose", action="store_true", help="Verbose gravity Newton iteration log.")

    args = parser.parse_args()

    # Normalize aliases
    if bool(args.gravity):
        args.state = "gravity"

    integrator = str(args.integrator).lower().strip()
    state = str(args.state).lower().strip()

    outdir = Path("outputs") / f"problem6_{state}_{integrator}_{args.mass_mode}"
    outdir.mkdir(parents=True, exist_ok=True)

    model, meta = build_portal_elastic(
        H=3.0,
        L=6.0,
        E=25e9,
        I_col=0.09,
        I_beam=0.12,
        A_col=0.30,
        A_beam=0.30,
        P_gravity_total=1500e3,
        nseg=int(args.nseg),
        nlgeom=bool(args.nlgeom),
        mass_mode=str(args.mass_mode),
        include_rot_inertia=(not bool(args.no_rot_inertia)),
        explicit_mass_dt=(0.00025 if integrator == "explicit" else None),
    )

    if state == "gravity":
        r = solve_gravity_only(
            model,
            tol=float(args.gravity_tol),
            max_iter=int(args.gravity_max_iter),
            n_load_steps=int(args.gravity_steps),
            verbose=bool(args.gravity_verbose),
        )

        summary = [
            "Problem 6 gravity-only (elastic portal frame)",
            f"ux_roof={r.get('ux_roof', float('nan')):.6e}",
            f"uy_roof={r.get('uy_roof', float('nan')):.6e}",
            f"drift={r.get('drift', float('nan')):.6e}",
            f"Vb={r.get('Vb', float('nan')):.6e}",
            f"mass_mode={meta.get('mass_mode')}",
            f"include_rot_inertia={meta.get('include_rot_inertia')}",
            f"M_total={meta.get('M_total', float('nan')):.6e}",
            f"mass_ux_total={meta.get('mass_ux_total', float('nan')):.6e}",
            f"mass_uy_total={meta.get('mass_uy_total', float('nan')):.6e}",
            f"inertia_th_total={meta.get('inertia_th_total', float('nan')):.6e}",
        ]
        (outdir / "problem6_gravity_summary.txt").write_text("\n".join(summary), encoding="utf-8")

        # Gravity plots: State 1 undeformed, State 2 static equilibrium
        try:
            u_g = r.get("u", None)
            if isinstance(u_g, np.ndarray):
                u_hist = np.vstack([np.zeros_like(u_g), u_g])
                drift_hist = np.array([0.0, float(r.get("drift", 0.0))], float)
                last_grav = {"u": u_hist, "t": np.array([0.0, 1.0], float), "drift": drift_hist, "snapshot_limit": float(args.snapshot_limit)}
                plot_structure_states(
                    model,
                    last_grav,
                    drift_height=float(meta.get("H", 3.0)),
                    snapshot_limit=float(args.snapshot_limit),
                    outfile="step1_gravity_states_members.png",
                    field="both",
                )
                plot_structure_states(
                    model,
                    last_grav,
                    drift_height=float(meta.get("H", 3.0)),
                    snapshot_limit=float(args.snapshot_limit),
                    outfile="step1_gravity_states_U.png",
                    field="u",
                )
                plot_structure_states(
                    model,
                    last_grav,
                    drift_height=float(meta.get("H", 3.0)),
                    snapshot_limit=float(args.snapshot_limit),
                    outfile="step1_gravity_states_S.png",
                    field="s",
                )
        except Exception as e:
            (outdir / "step1_gravity_plot_error.txt").write_text(str(e), encoding="utf-8")

        info = build_run_info(job="problem6_portico_elastico", output_dir=str(outdir), meta=meta)
        write_run_info(outdir, base_name="problem6_runinfo", info=info)

        print(f"[problema6] Gravity-only outputs in: {outdir.resolve()}")
        return

    # Dynamic run (single amplitude)
    out = run_one(
        integrator=integrator,
        model=model,
        meta=meta,
        A_g=float(args.A_g),
        t_end=float(args.t_end),
        dt=float(args.dt),
        drift_limit=float(args.drift_limit),
        snapshot_limit=float(args.snapshot_limit),
        alpha=float(args.alpha),
    )

    # Plot and summary (reuse existing helpers)
    plot_problem6_response(
        outdir,
        t=out["t"],
        drift=out["drift"],
        Vb=out["Vb"],
        snapshots=out.get("snapshots", {}),
    )

    # Structure plots for full history
    try:
        plot_structure_states(
            model,
            out,
            drift_height=float(meta.get("H", 3.0)),
            snapshot_limit=float(args.snapshot_limit),
            outfile="problem6_states_members.png",
            field="both",
        )
        plot_structure_states(
            model,
            out,
            drift_height=float(meta.get("H", 3.0)),
            snapshot_limit=float(args.snapshot_limit),
            outfile="problem6_states_U.png",
            field="u",
        )
        plot_structure_states(
            model,
            out,
            drift_height=float(meta.get("H", 3.0)),
            snapshot_limit=float(args.snapshot_limit),
            outfile="problem6_states_S.png",
            field="s",
        )
    except Exception as e:
        (outdir / "problem6_plot_error.txt").write_text(str(e), encoding="utf-8")

    lines = [
        "Problem 6 dynamic (elastic portal frame)",
        f"A_g={float(args.A_g):.3f} g",
        f"drift_max={float(np.max(out['drift'])):.6e}",
        f"Vb_max={float(np.max(np.abs(out['Vb']))):.6e}",
        f"integrator={integrator}",
        f"nlgeom={bool(args.nlgeom)}",
        f"mass_mode={meta.get('mass_mode')}",
        f"include_rot_inertia={meta.get('include_rot_inertia')}",
        f"dt={float(args.dt):.6e}",
        f"t_end={float(args.t_end):.6e}",
    ]
    (outdir / "problem6_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    info = build_run_info(job="problem6_portico_elastico", output_dir=str(outdir), meta=meta)
    write_run_info(outdir, base_name="problem6_runinfo", info=info)

    print(f"[problema6] OK. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()