"""Explicit time integration (Velocity Verlet).

This is intended as a simple, robust explicit scheme for educational / small models.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IncrementEnd, IncrementStart


# --- Constants ---
GRAVITY_TOL = 1e-10
GRAVITY_MAX_ITER = 80
EPS_REG = 1e-14


def _compute_drift(u: np.ndarray, node_indices: Tuple[int, int], height: float, model: Model) -> float:
    """Calculates inter-story drift ratio."""
    dof_1 = model.nodes[node_indices[0]].dof_u[0]
    dof_2 = model.nodes[node_indices[1]].dof_u[0]
    return 0.5 * (u[dof_1] + u[dof_2]) / height


def _solve_gravity_initialization(model: Model, nd: int, fd: np.ndarray) -> np.ndarray:
    """
    Performs a static implicit Newton-Raphson solve to equilibrate gravity 
    before starting the explicit dynamic time-stepping.
    """
    u = np.zeros(nd)
    u_free = u[fd].copy()
    nf = fd.size

    for _ in range(GRAVITY_MAX_ITER):
        u_trial = u.copy()
        u_trial[fd] = u_free
        
        model.update_column_yields(u_trial)
        K, Rint, _ = model.assemble(u_trial, u)
        
        # Residual = External - Internal
        res = model.load_const[fd] - Rint
        
        if np.linalg.norm(res) < GRAVITY_TOL * max(1.0, np.linalg.norm(model.load_const[fd])):
            model.commit()
            return u_trial
        
        # Robust linear solve
        mat = K + EPS_REG * np.eye(nf)
        try:
            du = np.linalg.solve(mat, res)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(mat, res, rcond=None)[0]
        
        u_free += du

    raise RuntimeError("Gravity initialization step failed to converge (Explicit Init).")


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
    reporter: Optional[Callable[[Any], None]] = None,
    step_id: int = 1,
) -> Dict[str, Any]:
    """
    Velocity Verlet / central-difference style explicit integration.
    
    Notes:
        - Uses lumped mass (model.mass_diag) and diagonal damping (model.C_diag).
        - No global stiffness matrix assembly or inversion is performed during the loop.
        - Conditional Stability: dt must be smaller than the critical time step.
    """
    solve_start = time.perf_counter()

    model.reset_state()
    
    # --- 1. Setup ---
    nd = model.ndof()
    fd = model.free_dofs()
    
    # Boolean mask for Free DOFs for efficient indexing
    # (Model returns reduced vectors, but Mass/Damping are full size)
    fd_mask = np.zeros(nd, dtype=bool)
    fd_mask[fd] = True

    M_diag = model.mass_diag
    C_diag = model.C_diag
    dt = float(t[1] - t[0])

    if verbose:
        print(f"[EXPLICIT] steps={t.size - 1} dt={dt:.6f}s (Velocity Verlet)")

    # Validate Mass
    # In explicit methods, M must be invertible (diagonal > 0) for all free DOFs.
    if np.any(M_diag[fd] <= 0.0):
        raise ValueError("Explicit integration requires positive mass for all free DOFs.")

    # Influence vector
    r = np.zeros(nd)
    r[M_diag > 0.0] = 1.0

    # History Arrays
    n_steps = t.size
    u_hist = np.zeros((n_steps, nd))
    v_hist = np.zeros((n_steps, nd))
    a_hist = np.zeros((n_steps, nd))
    drift_hist = np.zeros(n_steps)
    vb_hist = np.zeros(n_steps)
    iters_hist = np.zeros(n_steps - 1, dtype=int) # Always 0 for explicit
    hinge_hist = []

    if load_hist is None:
        load_hist = np.zeros((n_steps, nd))

    # --- 2. Initialization ---
    if u0 is None and v0 is None:
        u_n = _solve_gravity_initialization(model, nd, fd)
        v_n = np.zeros(nd)
    else:
        u_n = np.zeros(nd) if u0 is None else u0.copy()
        v_n = np.zeros(nd) if v0 is None else v0.copy()

    # Calculate initial acceleration a_0 = M^-1 * (P - I - C*v)
    model.update_column_yields(u_n)
    Rint0 = model.internal_force(u_n)
    p0 = model.load_const + load_hist[0] - M_diag * r * ag[0]
    
    a_n = np.zeros(nd)
    mass_mask = (M_diag > 0.0) & fd_mask
    
    a_n[mass_mask] = (
        p0[mass_mask] - Rint0[mass_mask] - C_diag[mass_mask] * v_n[mass_mask]
    ) / M_diag[mass_mask]

    # Store State 0
    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift_hist[0] = _compute_drift(u_n, drift_nodes, drift_height, model)
    vb_hist[0] = model.base_shear(u_n, base_nodes=base_nodes)

    snapshot_idx: Optional[int] = None
    
    # --- 3. Time Stepping Loop ---
    
    for n in range(n_steps - 1):
        model.update_column_yields(u_n)

        if reporter:
            reporter(IncrementStart(step_id=step_id, inc=n + 1, attempt=1, dt=dt))

        # --- A. First Half-Step (Predictor) ---
        # v_half = v_n + dt/2 * a_n
        v_half = v_n + 0.5 * dt * a_n
        
        # u_{n+1} = u_n + dt * v_half
        u_np1 = u_n + dt * v_half

        # --- B. Internal Force Evaluation ---
        # Note: assemble returns forces on Free DOFs only
        _, Rint_f, inf = model.assemble(u_np1, u_n)
        
        Rint_np1 = np.zeros(nd)
        Rint_np1[fd] = Rint_f

        # --- C. Second Half-Step (Corrector) ---
        p_np1 = model.load_const + load_hist[n + 1] - M_diag * r * ag[n + 1]

        # Solve M * a_{n+1} = P_{n+1} - R_{int, n+1} - C * v_{n+1/2}
        # Note: Using v_half for damping is a standard explicit approximation 
        # to decouple the velocity update.
        a_np1 = np.zeros(nd)
        numerator = (
            p_np1[mass_mask] 
            - Rint_np1[mass_mask] 
            - C_diag[mass_mask] * v_half[mass_mask]
        )
        a_np1[mass_mask] = numerator / M_diag[mass_mask]

        # v_{n+1} = v_{n+1/2} + dt/2 * a_{n+1}
        v_np1 = v_half + 0.5 * dt * a_np1

        # --- D. Finalize Step ---
        model.commit()

        u_n, v_n, a_n = u_np1, v_np1, a_np1

        # Record History
        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        drift_hist[n + 1] = _compute_drift(u_n, drift_nodes, drift_height, model)
        vb_hist[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)
        hinge_hist.append(inf.get("hinges", []))

        # Snapshot Logic
        current_drift_abs = abs(drift_hist[n + 1])
        if snapshot_idx is None and current_drift_abs >= drift_snapshot:
            snapshot_idx = n + 1

        if reporter:
            _report_success(reporter, step_id, n, dt, t)

        # Collapse check
        if current_drift_abs >= drift_limit:
            # Slice results
            trim = n + 2
            u_hist, v_hist, a_hist = u_hist[:trim], v_hist[:trim], a_hist[:trim]
            drift_hist, vb_hist = drift_hist[:trim], vb_hist[:trim]
            t, ag = t[:trim], ag[:trim]
            iters_hist = iters_hist[:n + 1]
            break

    # --- 4. Post-Processing ---
    solve_time_s = time.perf_counter() - solve_start
    
    if snapshot_idx is None:
        snapshot_idx = int(np.argmax(np.abs(drift_hist))) if drift_hist.size else -1
        
    is_valid_snap = 0 <= snapshot_idx < drift_hist.size
    snap_drift = float(drift_hist[int(snapshot_idx)]) if is_valid_snap else float("nan")
    snap_t = float(t[int(snapshot_idx)]) if is_valid_snap else float("nan")

    return {
        "t": t,
        "ag": ag,
        "u": u_hist,
        "v": v_hist,
        "a": a_hist,
        "drift": drift_hist,
        "Vb": vb_hist,
        "iters": iters_hist,
        "hinges": hinge_hist,
        "dt": dt,
        "n_steps": iters_hist.size,
        "iters_total": 0, # Explicit has 0 iterations
        "solve_time_s": solve_time_s,
        "snapshot_idx": int(snapshot_idx),
        "snapshot_t": snap_t,
        "snapshot_drift": snap_drift,
        "snapshot_limit": drift_snapshot,
        "snapshot_reached": (abs(snap_drift) >= drift_snapshot) if np.isfinite(snap_drift) else False,
    }


def _report_success(reporter, step_id, n, dt, t):
    """Helper for reporting successful explicit step."""
    reporter(
        IncrementEnd(
            step_id=step_id,
            inc=n + 1,
            attempt=1,
            converged=True,
            n_equil_iters=0,
            n_severe_iters=0,
            dt_completed=dt,
            step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
            step_time_completed=float(t[n + 1]),
            total_time_completed=float(t[n + 1]),
        )
    )