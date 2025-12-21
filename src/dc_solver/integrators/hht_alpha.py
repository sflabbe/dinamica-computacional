"""HHT-alpha integration with Newton iterations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Any

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IncrementEnd, IncrementStart, IterationReport


# --- Constants ---
EPS_REG = 1e-14  # Regularization for singular stiffness matrices
GRAVITY_TOL = 1e-10
GRAVITY_MAX_ITER = 60


@dataclass(frozen=True)
class HHTCoefficients:
    """Holds derived coefficients for the HHT-alpha method."""
    alpha: float
    beta: float
    gamma: float
    a0: float
    a1: float
    dt: float

    @classmethod
    def from_params(cls, alpha: float, dt: float) -> HHTCoefficients:
        if not (-1.0 / 3.0 - 1e-12 <= alpha <= 1e-12):
            raise ValueError("HHT-alpha requires alpha in [-1/3, 0].")
        
        gamma = 0.5 - alpha
        beta = 0.25 * (1.0 - alpha) ** 2
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        return cls(alpha, beta, gamma, a0, a1, dt)


def _compute_drift(u: np.ndarray, node_indices: Tuple[int, int], height: float, model: Model) -> float:
    """Calculates inter-story drift ratio."""
    dof_1 = model.nodes[node_indices[0]].dof_u[0]
    dof_2 = model.nodes[node_indices[1]].dof_u[0]
    # Average horizontal displacement of the two nodes divided by height
    return 0.5 * (u[dof_1] + u[dof_2]) / height


def _solve_gravity_initialization(
    model: Model, 
    nd: int, 
    fd: np.ndarray
) -> np.ndarray:
    """
    Performs a static analysis to resolve initial gravity loads 
    before the dynamic analysis begins.
    """
    u = np.zeros(nd)
    u_free = u[fd].copy()
    nf = fd.size

    for _ in range(GRAVITY_MAX_ITER):
        u_trial = u.copy()
        u_trial[fd] = u_free
        
        model.update_column_yields(u_trial)
        K, Rint, _ = model.assemble(u_trial, u)
        
        # Residual = External Loads - Internal Forces
        res = model.load_const[fd] - Rint
        
        # Check convergence
        load_norm = np.linalg.norm(model.load_const[fd])
        if np.linalg.norm(res) < GRAVITY_TOL * max(1.0, load_norm):
            model.commit()
            return u_trial

        # Solve for increment
        du = np.linalg.solve(K + EPS_REG * np.eye(nf), res)
        u_free += du

    raise RuntimeError("Gravity initialization step failed to converge.")


def hht_alpha_newton(
    model: Model,
    t: np.ndarray,
    ag: np.ndarray,
    drift_height: float,
    base_nodes: Tuple[int, int],
    drift_nodes: Tuple[int, int],
    drift_limit: float = 0.10,
    drift_snapshot: float = 0.04,
    alpha: float = -0.05,
    max_iter: int = 40,
    tol: float = 1e-6,
    verbose: bool = False,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    load_hist: Optional[np.ndarray] = None,
    reporter: Optional[Callable[[Any], None]] = None,
    step_id: int = 1,
) -> Dict[str, Any]:
    
    solve_start = time.perf_counter()
    model.reset_state()
    
    # --- 1. Setup & Pre-calculation ---
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size
    
    dt = float(t[1] - t[0])
    hht = HHTCoefficients.from_params(alpha, dt)

    # Mass and Damping diagonals
    M_diag = model.mass_diag
    C_diag = model.C_diag
    
    # Influence vector (1.0 for mass DOFs)
    r = np.zeros(nd)
    r[M_diag > 0.0] = 1.0

    # Initialize History Arrays
    n_steps = t.size
    u_hist = np.zeros((n_steps, nd))
    v_hist = np.zeros((n_steps, nd))
    a_hist = np.zeros((n_steps, nd))
    drift_hist = np.zeros(n_steps)
    vb_hist = np.zeros(n_steps)
    iters_hist = np.zeros(n_steps - 1, dtype=int)
    hinge_hist = []

    # Handle Load History
    if load_hist is None:
        load_hist = np.zeros((n_steps, nd))

    # --- 2. Initial State Determination ---
    if u0 is None and v0 is None:
        # Static Gravity Solve
        u_n = _solve_gravity_initialization(model, nd, fd)
        v_n = np.zeros(nd)
        # For a static start, acceleration is zero unless external forces are unbalanced immediately
        a_n = np.zeros(nd) 
    else:
        # Continuation or custom start
        u_n = np.zeros(nd) if u0 is None else u0.copy()
        v_n = np.zeros(nd) if v0 is None else v0.copy()
        
        # Calculate initial acceleration based on Equation of Motion
        model.update_column_yields(u_n)
        Rint0 = model.internal_force(u_n)
        
        p0 = model.load_const.copy()
        if load_hist.shape[0] > 0:
            p0 += load_hist[0]
        
        # P - I - C*v = M*a  =>  a = M^-1 * (P - I - C*v)
        # External load includes earthquake acceleration term: - M * r * ag
        p_dynamic = p0 - M_diag * r * ag[0]
        
        a_n = np.zeros(nd)
        mass_mask = M_diag > 0.0
        numerator = p_dynamic[mass_mask] - Rint0[mass_mask] - C_diag[mass_mask] * v_n[mass_mask]
        a_n[mass_mask] = numerator / M_diag[mass_mask]

    # Store State 0
    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift_hist[0] = _compute_drift(u_n, drift_nodes, drift_height, model)
    vb_hist[0] = model.base_shear(u_n, base_nodes=base_nodes)

    snapshot_idx: Optional[int] = None
    
    # --- 3. Time Stepping Loop ---
    p_const = model.load_const.copy()
    
    for n in range(n_steps - 1):
        # Update model properties based on previous converged state
        model.update_column_yields(u_n)

        # Calculate effective loads for HHT
        p_n = p_const + load_hist[n] - M_diag * r * ag[n]
        p_np1 = p_const + load_hist[n + 1] - M_diag * r * ag[n + 1]
        
        # HHT-alpha effective load vector
        p_alpha = (1.0 + hht.alpha) * p_np1 - hht.alpha * p_n

        # Internal force at previous step (for HHT balance)
        _, Rint_n, _ = model.assemble(u_n, u_n)

        # Predictor Step
        u_pred = u_n + dt * v_n + dt * dt * (0.5 - hht.beta) * a_n
        v_pred = v_n + dt * (1.0 - hht.gamma) * a_n

        # Initialize trial variables for Newton loop
        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy() # The "committed" state we are stepping FROM

        if reporter:
            reporter(IncrementStart(
                step_id=step_id, inc=n+1, attempt=1, dt=dt, 
                step_time=float(t[n]), total_time=float(t[n]), is_cutback_attempt=False
            ))

        # --- Newton-Raphson Loop ---
        converged = False
        final_inf = None
        
        for it in range(1, max_iter + 1):
            # Construct trial vector
            u_trial = u_comm_step.copy()
            u_trial[fd] = u_free

            # Get Tangent Stiffness and Internal Forces
            K_tan, Rint, inf = model.assemble(u_trial, u_comm_step)

            # Update kinematics based on HHT formulas
            # u_trial = u_pred + beta*dt^2 * a_trial  => solve for a_trial
            a_trial = hht.a0 * (u_trial - u_pred)
            v_trial = v_pred + (hht.gamma * dt) * a_trial

            # Calculate Residual
            # Res = P_alpha - ( (1+alpha)I(u) - alpha*I(u_n) + C*v + M*a )
            inertial_forces = M_diag[fd] * a_trial[fd]
            damping_forces = C_diag[fd] * v_trial[fd]
            restoring_forces = (1.0 + hht.alpha) * Rint - hht.alpha * Rint_n
            
            res = p_alpha[fd] - (restoring_forces + damping_forces + inertial_forces)

            # Check Convergence
            res_norm = float(np.linalg.norm(res))
            scale = 1.0 + np.linalg.norm(p_alpha[fd])
            
            if res_norm <= tol * scale:
                converged = True
                u_n, v_n, a_n = u_trial, v_trial, a_trial
                final_inf = inf
                iters_hist[n] = it
                model.commit() # Commit state to material models
                
                if reporter:
                    _report_convergence(reporter, step_id, n, it, dt, t, res_norm, True)
                break
            
            # Linear Solve for Correction
            # K_eff = (1+alpha)K_T + a0*M + a1*C
            K_eff = ((1.0 + hht.alpha) * K_tan 
                     + np.diag(M_diag[fd] * hht.a0) 
                     + np.diag(C_diag[fd] * hht.a1))

            # Handle potentially singular matrix
            mat = K_eff + EPS_REG * np.eye(nf)
            
            try:
                du = np.linalg.solve(mat, res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(mat, res, rcond=None)[0]

            u_free += du

            if reporter:
                _report_iteration(reporter, step_id, n, it, res_norm, res, fd, du)

        # --- End Newton Loop ---

        if not converged:
            if reporter:
                _report_convergence(reporter, step_id, n, max_iter, dt, t, 0.0, False)
            raise RuntimeError(f"No convergence in step {n+1} / t={t[n+1]:.3f}s (HHT-alpha).")

        # Record History
        hinge_hist.append(final_inf["hinges"] if final_inf else {})
        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        
        current_drift = _compute_drift(u_n, drift_nodes, drift_height, model)
        drift_hist[n + 1] = current_drift
        vb_hist[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)

        # Snapshot Logic
        if snapshot_idx is None and abs(current_drift) >= drift_snapshot:
            snapshot_idx = n + 1

        # Collapse Check
        if abs(current_drift) >= drift_limit:
            if verbose:
                print(f"COLLAPSE by drift >= {100*drift_limit:.1f}% at t={t[n+1]:.3f}s")
            # Trim arrays to valid data
            trim_idx = n + 2
            u_hist = u_hist[:trim_idx]
            v_hist = v_hist[:trim_idx]
            a_hist = a_hist[:trim_idx]
            drift_hist = drift_hist[:trim_idx]
            vb_hist = vb_hist[:trim_idx]
            t = t[:trim_idx]
            ag = ag[:trim_idx]
            iters_hist = iters_hist[: n + 1]
            break

    # --- 4. Post-Processing & Stats ---
    solve_time_s = time.perf_counter() - solve_start
    rho_inf = (1.0 + alpha) / (1.0 - alpha)

    # Handle snapshot stats
    if snapshot_idx is None:
        # If never reached snapshot limit, take the max drift index
        snapshot_idx = int(np.argmax(np.abs(drift_hist))) if drift_hist.size else -1
    
    # Safely extract snapshot values
    is_valid_snap = 0 <= snapshot_idx < drift_hist.size
    snap_drift = float(drift_hist[int(snapshot_idx)]) if is_valid_snap else float("nan")
    snap_t = float(t[int(snapshot_idx)]) if is_valid_snap else float("nan")
    snap_reached = (abs(snap_drift) >= drift_snapshot) if np.isfinite(snap_drift) else False

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
        "alpha": alpha,
        "gamma": hht.gamma,
        "beta": hht.beta,
        "rho_inf": rho_inf,
        "n_steps": iters_hist.size,
        "iters_total": int(np.sum(iters_hist)) if iters_hist.size else 0,
        "solve_time_s": solve_time_s,
        "snapshot_idx": int(snapshot_idx),
        "snapshot_t": snap_t,
        "snapshot_drift": snap_drift,
        "snapshot_limit": drift_snapshot,
        "snapshot_reached": snap_reached,
    }


# --- Reporting Helpers (To de-clutter main logic) ---

def _report_convergence(reporter, step_id, n, it, dt, t, res_norm, converged):
    """Helper to dispatch IncrementEnd reports."""
    reporter(
        IncrementEnd(
            step_id=step_id,
            inc=n + 1,
            attempt=1,
            converged=converged,
            n_equil_iters=it,
            n_severe_iters=0,
            dt_completed=dt,
            step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
            step_time_completed=float(t[n + 1]),
            total_time_completed=float(t[n + 1]),
        )
    )
    if converged:
        # Also report the final successful iteration
        reporter(
            IterationReport(
                step_id=step_id,
                inc=n + 1,
                attempt=1,
                it=it,
                residual_norm=res_norm,
                residual_max=0.0, # Simplified
                residual_dof=None,
                residual_node=None,
                residual_component_label="FORCE",
                correction_norm=0.0,
                correction_max=0.0,
                converged_force=True,
                converged_moment=True,
                note=None,
            )
        )

def _report_iteration(reporter, step_id, n, it, res_norm, res, fd, du):
    """Helper to dispatch IterationReport for non-converged steps."""
    res_max = float(np.max(np.abs(res))) if res.size else 0.0
    res_idx = int(np.argmax(np.abs(res))) if res.size else None
    res_dof = int(fd[res_idx]) if res_idx is not None else None
    
    reporter(
        IterationReport(
            step_id=step_id,
            inc=n + 1,
            attempt=1,
            it=it,
            residual_norm=res_norm,
            residual_max=res_max,
            residual_dof=res_dof,
            residual_node=None,
            residual_component_label="FORCE",
            correction_norm=float(np.linalg.norm(du)),
            correction_max=float(np.max(np.abs(du))) if du.size else 0.0,
            converged_force=False,
            converged_moment=False,
            note=None,
        )
    )