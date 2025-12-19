"""Explicit time integration (Velocity Verlet).

This is intended as a simple, robust explicit scheme for educational / small models.
It follows the same output contract as HHT/Newmark where possible.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IncrementEnd, IncrementStart


def explicit_verlet(
    model: Model,
    t: np.ndarray,
    ag: np.ndarray,
    drift_height: float,
    base_nodes: Tuple[int, int],
    drift_nodes: Tuple[int, int],
    drift_limit: float = 0.10,
    drift_snapshot: float = 0.04,
    verbose: bool = False,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    load_hist: Optional[np.ndarray] = None,
    reporter: Optional[Callable[[object], None]] = None,
    step_id: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Velocity Verlet / central-difference style explicit integration.

    Notes
    - Uses lumped mass: model.mass_diag and diagonal damping model.C_diag.
    - Hinge state is handled by calling model.assemble(u_trial, u_comm) each step
      and committing after acceptance (explicit scheme has no iterations).
    """
    solve_start = time.perf_counter()

    model.reset_state()
    nd = model.ndof()
    fd = model.free_dofs()

    # Initial equilibrium under constant loads unless u0/v0 provided
    if u0 is None and v0 is None:
        u = np.zeros(nd)
        u_free = u[fd].copy()
        nf = fd.size
        for _ in range(80):
            u_trial = u.copy()
            u_trial[fd] = u_free
            model.update_column_yields(u_trial)
            K, Rint, _ = model.assemble(u_trial, u)
            res = model.load_const[fd] - Rint
            if np.linalg.norm(res) < 1e-10 * max(1.0, np.linalg.norm(model.load_const[fd])):
                u = u_trial.copy()
                model.commit()
                break
            try:
                du = np.linalg.solve(K + 1e-14 * np.eye(nf), res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K + 1e-14 * np.eye(nf), res, rcond=None)[0]
            u_free += du
        else:
            raise RuntimeError("No converge el paso estático de gravedad (explicit init).")
        u_n = u.copy()
        v_n = np.zeros(nd)
    else:
        u_n = np.zeros(nd) if u0 is None else u0.copy()
        v_n = np.zeros(nd) if v0 is None else v0.copy()

    M = model.mass_diag
    C = model.C_diag
    dt = float(t[1] - t[0])

    # influence vector (same assumption as HHT/Newmark here)
    r = np.zeros(nd)
    r[np.where(M > 0.0)[0]] = 1.0

    if load_hist is None:
        load_hist = np.zeros((t.size, nd))

    # Initial acceleration from equilibrium at step 0
    model.update_column_yields(u_n)
    Rint0 = model.internal_force(u_n)
    p0 = model.load_const + load_hist[0] - M * r * ag[0]
    a_n = np.zeros(nd)
    mass_mask = M > 0.0
    a_n[mass_mask] = (p0[mass_mask] - Rint0[mass_mask] - C[mass_mask] * v_n[mass_mask]) / M[mass_mask]

    # History arrays
    u_hist = np.zeros((t.size, nd))
    v_hist = np.zeros((t.size, nd))
    a_hist = np.zeros((t.size, nd))
    drift = np.zeros(t.size)
    Vb = np.zeros(t.size)
    iters = np.zeros(t.size - 1, dtype=int)
    hinge_hist = []

    ux2 = model.nodes[drift_nodes[0]].dof_u[0]
    ux3 = model.nodes[drift_nodes[1]].dof_u[0]

    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift[0] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
    Vb[0] = model.base_shear(u_n, base_nodes=base_nodes)

    snapshot_idx = None
    snapshot_reached = False
    snapshot_t = float("nan")
    snapshot_drift = float("nan")

    if verbose:
        print(f"[EXPLICIT] steps={t.size - 1} dt={dt:.6f}s (Velocity Verlet)")

    for n in range(t.size - 1):
        model.update_column_yields(u_n)

        if reporter is not None:
            reporter(IncrementStart(step_id=step_id, inc=n + 1, attempt=1, dt=float(dt)))

        # Velocity Verlet:
        v_half = v_n + 0.5 * dt * a_n
        u_np1 = u_n + dt * v_half

        # Evaluate internal forces (trial) and commit hinge state at u_{n+1}
        K_tan, Rint_np1, inf = model.assemble(u_np1, u_n)

        p_np1 = model.load_const + load_hist[n + 1] - M * r * ag[n + 1]

        a_np1 = np.zeros(nd)
        # damping with v_half is a reasonable explicit approximation
        a_np1[mass_mask] = (p_np1[mass_mask] - Rint_np1[mass_mask] - C[mass_mask] * v_half[mass_mask]) / M[mass_mask]
        v_np1 = v_half + 0.5 * dt * a_np1

        # accept step and commit trial hinge state
        model.commit()

        u_n = u_np1
        v_n = v_np1
        a_n = a_np1

        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        drift[n + 1] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
        Vb[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)
        iters[n] = 0
        hinge_hist.append(inf.get("hinges", []))

        # snapshot logic
        if (not snapshot_reached) and abs(drift[n + 1]) >= drift_snapshot:
            snapshot_reached = True
            snapshot_idx = n + 1
            snapshot_t = float(t[n + 1])
            snapshot_drift = float(drift[n + 1])

        if reporter is not None:
            reporter(
                IncrementEnd(
                    step_id=step_id,
                    inc=n + 1,
                    attempt=1,
                    converged=True,
                    n_equil_iters=0,
                    n_severe_iters=0,
                    dt_completed=float(dt),
                    step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
                    step_time_completed=float(t[n + 1]),
                    total_time_completed=float(t[n + 1]),
                )
            )

        if abs(drift[n + 1]) >= drift_limit:
            break

    solve_time_s = time.perf_counter() - solve_start
    if snapshot_idx is None:
        snapshot_idx = int(np.argmax(np.abs(drift)))
        snapshot_t = float(t[snapshot_idx])
        snapshot_drift = float(drift[snapshot_idx])

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
        "n_steps": int(t.size - 1),
        "iters_total": int(np.sum(iters)),
        "solve_time_s": float(solve_time_s),
        "snapshot_idx": int(snapshot_idx),
        "snapshot_t": float(snapshot_t),
        "snapshot_drift": float(snapshot_drift),
        "snapshot_limit": float(drift_snapshot),
        "snapshot_reached": bool(snapshot_reached),
    }
