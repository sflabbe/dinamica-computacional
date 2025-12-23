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


def estimate_explicit_dt_crit(model: Model, u_ref: Optional[np.ndarray] = None) -> float:
    """Estimate a critical explicit time step for the current model.

    Uses the largest eigenvalue of M^{-1}K (free DOFs) computed from a tangent
    stiffness K assembled at the reference configuration u_ref.

    Returns
    -------
    dt_crit : float
        Approximate critical step size for central difference / Velocity-Verlet
        integration: dt_crit ≈ 2 / ω_max.
    """
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size
    if nf == 0:
        return float("inf")

    if u_ref is None:
        u_ref = np.zeros(nd, dtype=float)

    # Assemble tangent at the reference state
    K, _, _ = model.assemble(u_ref, u_ref)

    M = np.asarray(model.mass_diag[fd], dtype=float)
    # Avoid divide-by-zero (constrained DOFs are removed already)
    M = np.where(M > 0.0, M, 1.0)

    # Form symmetric matrix A = D^{-1/2} K D^{-1/2} whose eigenvalues are ω^2
    inv_sqrt_M = 1.0 / np.sqrt(M)
    A = (inv_sqrt_M[:, None] * K) * inv_sqrt_M[None, :]
    A = 0.5 * (A + A.T)  # numerical symmetry
    try:
        lam = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        return float("nan")

    lam_max = float(np.max(lam)) if lam.size else 0.0
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return float("inf")

    omega_max = float(np.sqrt(lam_max))
    return 2.0 / omega_max


def _compute_drift(
    u: np.ndarray,
    roof_nodes: Tuple[int, int],
    height: float,
    model: Model,
    base_nodes: Optional[Tuple[int, int]] = None,
) -> float:
    """Inter-story drift ratio = (roof_avg_ux - base_avg_ux) / height.

    Using base_nodes makes the definition robust if support motion is introduced later.
    """
    ux_r1 = model.nodes[roof_nodes[0]].dof_u[0]
    ux_r2 = model.nodes[roof_nodes[1]].dof_u[0]
    roof = 0.5 * (u[ux_r1] + u[ux_r2])

    base = 0.0
    if base_nodes is not None:
        ux_b1 = model.nodes[base_nodes[0]].dof_u[0]
        ux_b2 = model.nodes[base_nodes[1]].dof_u[0]
        base = 0.5 * (u[ux_b1] + u[ux_b2])

    return float(roof - base) / float(height)


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

    # Reset only for a fresh start (no provided initial conditions).
    if u0 is None and v0 is None:
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

    # Validate Mass
    # In explicit methods, M must be invertible (diagonal > 0) for all free DOFs.
    if np.any(M_diag[fd] <= 0.0):
        raise ValueError("Explicit integration requires positive mass for all free DOFs.")

    # Influence vector
    r = np.zeros(nd)
    # Influence vector: horizontal ground acceleration only (ux DOFs)
    for node in model.nodes:
        r[node.dof_u[0]] = 1.0
    # History Arrays
    n_steps = t.size
    u_hist = np.zeros((n_steps, nd))
    v_hist = np.zeros((n_steps, nd))
    a_hist = np.zeros((n_steps, nd))
    drift_hist = np.zeros(n_steps)
    vb_hist = np.zeros(n_steps)
    iters_hist = np.zeros(n_steps - 1, dtype=int) # Always 0 for explicit
    hinge_hist = []

    # --- Energy balance accumulators (cumulative works + kinetic) ---
    T_hist = np.zeros(n_steps)
    Wext_hist = np.zeros(n_steps)
    Wint_hist = np.zeros(n_steps)
    Wdamp_hist = np.zeros(n_steps)
    Wpl_hist = np.zeros(n_steps)
    res_hist = np.zeros(n_steps)
    p_prev = None  # type: ignore
    R_prev = None  # type: ignore
    v_prev = None  # type: ignore
    Rint_final = None  # type: ignore



    if load_hist is None:
        load_hist = np.zeros((n_steps, nd))

    # --- 2. Initialization ---
    if u0 is None and v0 is None:
        u_n = _solve_gravity_initialization(model, nd, fd)
        v_n = np.zeros(nd)
    else:
        u_n = np.zeros(nd) if u0 is None else u0.copy()
        v_n = np.zeros(nd) if v0 is None else v0.copy()

    # --- 2b. Stability / substepping (keeps the external time grid intact) ---
    dt_crit = estimate_explicit_dt_crit(model, u_ref=u_n)
    safety = 0.90
    n_sub = 1
    if np.isfinite(dt_crit) and dt_crit > 0.0:
        n_sub = int(np.ceil(dt / (safety * dt_crit)))
        n_sub = max(1, n_sub)
    dt_sub = dt / float(n_sub)

    if verbose:
        msg = f"[EXPLICIT] steps={t.size - 1} dt={dt:.6f}s"
        if np.isfinite(dt_crit):
            msg += f" dt_crit~{dt_crit:.6e}s n_sub={n_sub} dt_sub={dt_sub:.6e}s"
        msg += " (Velocity Verlet)"
        print(msg)

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
    drift_hist[0] = _compute_drift(u_n, drift_nodes, drift_height, model, base_nodes)
    vb_hist[0] = model.base_shear(u_n, base_nodes=base_nodes)

    # Energy balance init at step 0 (free DOFs only)
    try:
        model.update_column_yields(u_n)
        _, R0, _ = model.assemble(u_n, u_n)
        p0 = p_const + load_hist[0] - M_diag * r * ag[0]
        T_hist[0] = 0.5 * float(np.sum(M_diag[fd] * (v_n[fd] ** 2)))
        p_prev = p0[fd].copy()
        R_prev = R0.copy()
        v_prev = v_n[fd].copy()
    except Exception:
        pass

    snapshot_idx: Optional[int] = None
    
    # --- 3. Time Stepping Loop ---
    
    for n in range(n_steps - 1):
        if reporter:
            reporter(IncrementStart(step_id=step_id, inc=n + 1, attempt=1, dt=dt))

        inf_last: Dict[str, Any] = {}

        # Substep integration over [t_n, t_{n+1}] using linear interpolation
        # of the input record (ag) and optional external load history.
        for j in range(n_sub):
            model.update_column_yields(u_n)

            frac = float(j + 1) / float(n_sub)
            ag_end = float(ag[n] + frac * (ag[n + 1] - ag[n]))

            if load_hist is None:
                load_end = np.zeros(nd)
            else:
                load_end = load_hist[n] + frac * (load_hist[n + 1] - load_hist[n])

            # --- A. First Half-Step (Predictor) ---
            v_half = v_n + 0.5 * dt_sub * a_n
            u_np1 = u_n + dt_sub * v_half

            # --- B. Internal Force Evaluation ---
            _, Rint_f, inf_last = model.assemble(u_np1, u_n)
            Rint_final = Rint_f

            Rint_np1 = np.zeros(nd)
            Rint_np1[fd] = Rint_f

            # --- C. Second Half-Step (Corrector) ---
            p_end = model.load_const + load_end - M_diag * r * ag_end
            a_np1 = np.zeros(nd)
            numerator = (
                p_end[mass_mask]
                - Rint_np1[mass_mask]
                - C_diag[mass_mask] * v_half[mass_mask]
            )
            a_np1[mass_mask] = numerator / M_diag[mass_mask]
            v_np1 = v_half + 0.5 * dt_sub * a_np1

            model.commit()
            u_n, v_n, a_n = u_np1, v_np1, a_np1

        # Record History
        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n

        # Energy balance update
        try:
            du = (u_hist[n + 1][fd] - u_hist[n][fd])
            p_curr = (p_const + load_hist[n + 1] - M_diag * r * ag[n + 1])[fd]
            R_curr = Rint_final
            if p_prev is None:
                p_prev = (p_const + load_hist[n] - M_diag * r * ag[n])[fd]
            if R_prev is None:
                _, R_prev0, _ = model.assemble(u_hist[n], u_hist[n])
                R_prev = R_prev0
            if v_prev is None:
                v_prev = v_hist[n][fd]
            dWext = 0.5 * float(np.dot((p_prev + p_curr), du))
            dWint = 0.5 * float(np.dot((R_prev + R_curr), du))
            pow_d0 = float(np.sum(C_diag[fd] * (v_prev ** 2)))
            v_curr = v_hist[n + 1][fd]
            pow_d1 = float(np.sum(C_diag[fd] * (v_curr ** 2)))
            dWd = 0.5 * (pow_d0 + pow_d1) * float(dt)
            Wext_hist[n + 1] = Wext_hist[n] + dWext
            Wint_hist[n + 1] = Wint_hist[n] + dWint
            Wdamp_hist[n + 1] = Wdamp_hist[n] + dWd
            dWpl = 0.0
            if isinstance(inf_last, dict) and isinstance(inf_last.get('hinges', None), list):
                for hh in inf_last.get('hinges', []):
                    try:
                        dWpl += float(hh.get('dW_pl', 0.0))
                    except Exception:
                        pass
            Wpl_hist[n + 1] = Wpl_hist[n] + float(dWpl)
            T_hist[n + 1] = 0.5 * float(np.sum(M_diag[fd] * (v_hist[n + 1][fd] ** 2)))
            res_hist[n + 1] = (T_hist[n + 1] - T_hist[0]) + Wint_hist[n + 1] + Wdamp_hist[n + 1] - Wext_hist[n + 1]
            p_prev = p_curr.copy()
            R_prev = np.array(R_curr, dtype=float).copy()
            v_prev = np.array(v_curr, dtype=float).copy()
        except Exception:
            pass
        drift_hist[n + 1] = _compute_drift(u_n, drift_nodes, drift_height, model, base_nodes)
        vb_hist[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)
        hinge_hist.append(inf_last.get("hinges", []))

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
        "energy": {
            "T": T_hist,
            "W_ext": Wext_hist,
            "W_int": Wint_hist,
            "W_damp": Wdamp_hist,
            "W_pl": Wpl_hist,
            "residual": res_hist,
        },
        "dt": dt,
        "dt_sub": float(dt_sub),
        "dt_crit_est": float(dt_crit) if np.isfinite(dt_crit) else float("nan"),
        "n_substeps": int(n_sub),
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