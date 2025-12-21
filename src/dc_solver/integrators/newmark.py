"""Newmark-beta integration with Newton iterations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IncrementEnd, IncrementStart, IterationReport


# --- Constants ---
EPS_REG = 1e-14
GRAVITY_TOL = 1e-10
GRAVITY_MAX_ITER = 60


@dataclass(frozen=True)
class NewmarkCoefficients:
    """derived coefficients for Newmark-beta method."""
    beta: float
    gamma: float
    a0: float  # Coeff for Mass in K_eff: 1 / (beta * dt^2)
    a1: float  # Coeff for Damping in K_eff: gamma / (beta * dt)
    dt: float

    @classmethod
    def from_params(cls, beta: float, gamma: float, dt: float) -> NewmarkCoefficients:
        # Standard Newmark constants
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        return cls(beta, gamma, a0, a1, dt)


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
    """Performs static gravity initialization."""
    u = np.zeros(nd)
    u_free = u[fd].copy()
    nf = fd.size

    for _ in range(GRAVITY_MAX_ITER):
        u_trial = u.copy()
        u_trial[fd] = u_free
        
        model.update_column_yields(u_trial)
        K, Rint, _ = model.assemble(u_trial, u)
        
        res = model.load_const[fd] - Rint
        
        load_norm = np.linalg.norm(model.load_const[fd])
        if np.linalg.norm(res) < GRAVITY_TOL * max(1.0, load_norm):
            model.commit()
            return u_trial

        mat = K + EPS_REG * np.eye(nf)
        try:
            du = np.linalg.solve(mat, res)
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(mat, res, rcond=None)[0]
        u_free += du

    raise RuntimeError("Gravity initialization step failed to converge.")


def newmark_beta_newton(
    model: Model,
    t: np.ndarray,
    ag: np.ndarray,
    drift_height: float,
    base_nodes: Tuple[int, int],
    drift_nodes: Tuple[int, int],
    drift_limit: float = 0.10,
    drift_snapshot: float = 0.04,
    beta: float = 0.25,
    gamma: float = 0.50,
    max_iter: int = 40,
    tol: float = 1e-6,
    verbose: bool = False,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    load_hist: Optional[np.ndarray] = None,
    reporter: Optional[Callable[[Any], None]] = None,
    step_id: int = 1,
) -> Dict[str, Any]:
    """
    Constant-average-acceleration Newmark method (implicit) with Newton iterations.
    """
    solve_start = time.perf_counter()

    model.reset_state()
    
    # --- 1. Setup & Pre-calculation ---
    nd = model.ndof()
    fd = model.free_dofs()
    fd_mask = np.zeros(nd, dtype=bool)
    fd_mask[fd] = True
    nf = fd.size
    
    dt = float(t[1] - t[0])
    nmk = NewmarkCoefficients.from_params(beta, gamma, dt)

    if verbose:
        print(f"[NEWMARK] steps={t.size - 1} dt={dt:.6f}s beta={beta:.3f} gamma={gamma:.3f}")

    M_diag = model.mass_diag
    C_diag = model.C_diag

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
    iters_hist = np.zeros(n_steps - 1, dtype=int)
    hinge_hist = []

    if load_hist is None:
        load_hist = np.zeros((n_steps, nd))

    # --- 2. Initial State Determination ---
    if u0 is None and v0 is None:
        u_n = _solve_gravity_initialization(model, nd, fd)
        v_n = np.zeros(nd)
        a_n = np.zeros(nd)
    else:
        u_n = np.zeros(nd) if u0 is None else u0.copy()
        v_n = np.zeros(nd) if v0 is None else v0.copy()
        
        # Calculate initial acceleration
        model.update_column_yields(u_n)
        Rint0 = model.internal_force(u_n)
        p0 = model.load_const.copy()
        if load_hist.shape[0] > 0:
            p0 += load_hist[0]
        
        p_dynamic = p0 - M_diag * r * ag[0]
        
        a_n = np.zeros(nd)
        mass_mask = (M_diag > 0.0) & fd_mask
        numerator = p_dynamic[mass_mask] - Rint0[mass_mask] - C_diag[mass_mask] * v_n[mass_mask]
        a_n[mass_mask] = numerator / M_diag[mass_mask]

    # Store Step 0
    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift_hist[0] = _compute_drift(u_n, drift_nodes, drift_height, model, base_nodes)
    vb_hist[0] = model.base_shear(u_n, base_nodes=base_nodes)

    snapshot_idx: Optional[int] = None
    p_const = model.load_const.copy()

    # --- 3. Time Stepping Loop ---
    

    for n in range(n_steps - 1):
        model.update_column_yields(u_n)

        # External Force at t + dt
        p_np1 = p_const + load_hist[n + 1] - M_diag * r * ag[n + 1]

        # --- Predictor Phase ---
        # u_pred = u + dt*v + dt^2/2 * (1-2beta)*a
        u_pred = u_n + dt * v_n + 0.5 * dt * dt * (1.0 - 2.0 * nmk.beta) * a_n
        v_pred = v_n + dt * (1.0 - nmk.gamma) * a_n

        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy()

        if reporter:
            reporter(IncrementStart(step_id=step_id, inc=n + 1, attempt=1, dt=dt))

        converged = False
        final_inf = {"hinges": []}

        # --- Corrector (Newton) Phase ---
        for it in range(max_iter):
            u_trial = u_comm_step.copy()
            u_trial[fd] = u_free

            K_tan, Rint, inf = model.assemble(u_trial, u_comm_step)
            final_inf = inf

            # Update kinematics based on Newmark formulas
            # a = a0 * (u - u_pred)
            # v = v_pred + gamma*dt*a
            a_trial = nmk.a0 * (u_trial - u_pred)
            v_trial = v_pred + (nmk.gamma * dt) * a_trial

            # Residual = P_ext - I(u) - M*a - C*v
            inertial = M_diag[fd] * a_trial[fd] # Using diagonal matrix multiplication
            damping = C_diag[fd] * v_trial[fd]
            
            res = p_np1[fd] - Rint - inertial - damping
            
            # Check Convergence
            res_norm = float(np.linalg.norm(res))
            scale = max(1.0, np.linalg.norm(p_np1[fd]))
            
            if res_norm < tol * scale:
                u_n, v_n, a_n = u_trial, v_trial, a_trial
                iters_hist[n] = it
                converged = True
                model.commit()
                if reporter:
                    _report_convergence(reporter, step_id, n, it, dt, t, res_norm, True)
                break

            # Calculate Effective Stiffness
            # K_eff = K_t + a0*M + a1*C
            K_eff = (K_tan 
                     + np.diag(M_diag[fd] * nmk.a0) 
                     + np.diag(C_diag[fd] * nmk.a1))

            # Solve linear system
            mat = K_eff + EPS_REG * np.eye(nf)
            try:
                du = np.linalg.solve(mat, res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(mat, res, rcond=None)[0]
            
            u_free += du
            
            # Report iteration if needed
            if reporter: 
                 _report_iteration(reporter, step_id, n, it, res_norm, res, fd, du)
        
        # --- End Newton Loop ---

        hinge_hist.append(final_inf.get("hinges", []))

        if not converged:
            if reporter:
                _report_convergence(reporter, step_id, n, max_iter, dt, t, 0.0, False)
            raise RuntimeError(f"Newmark convergence failed at step {n+1}, t={t[n+1]:.3f}s")

        # Record History
        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        
        current_drift = _compute_drift(u_n, drift_nodes, drift_height, model, base_nodes)
        drift_hist[n + 1] = current_drift
        vb_hist[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)

        # Snapshot Logic
        if snapshot_idx is None and abs(current_drift) >= drift_snapshot:
            snapshot_idx = n + 1

        if abs(current_drift) >= drift_limit:
            if verbose:
                print(f"COLLAPSE by drift >= {100*drift_limit:.1f}% at t={t[n+1]:.3f}s")
            
            # Trim arrays
            trim = n + 2
            u_hist, v_hist, a_hist = u_hist[:trim], v_hist[:trim], a_hist[:trim]
            drift_hist, vb_hist = drift_hist[:trim], vb_hist[:trim]
            t, ag = t[:trim], ag[:trim]
            iters_hist = iters_hist[: n + 1]
            break

    # --- 4. Stats & Return ---
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
        "beta": beta,
        "gamma": gamma,
        "n_steps": iters_hist.size,
        "iters_total": int(np.sum(iters_hist)) if iters_hist.size else 0,
        "solve_time_s": solve_time_s,
        "snapshot_idx": int(snapshot_idx),
        "snapshot_t": snap_t,
        "snapshot_drift": snap_drift,
        "snapshot_limit": drift_snapshot,
        "snapshot_reached": (abs(snap_drift) >= drift_snapshot) if np.isfinite(snap_drift) else False,
    }


# --- Reporting Helpers ---

def _report_convergence(reporter, step_id, n, it, dt, t, res_norm, converged):
    reporter(
        IncrementEnd(
            step_id=step_id,
            inc=n + 1,
            attempt=1,
            converged=converged,
            n_equil_iters=it,
            n_severe_iters=0,
            dt_completed=float(dt),
            step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
            step_time_completed=float(t[n + 1]),
            total_time_completed=float(t[n + 1]),
        )
    )
    if converged:
        reporter(
            IterationReport(
                step_id=step_id, inc=n + 1, attempt=1, it=it,
                residual_norm=res_norm, residual_max=0.0, residual_dof=None,
                residual_node=None, residual_component_label="FORCE",
                correction_norm=0.0, correction_max=0.0,
                converged_force=True, converged_moment=True, note=None,
            )
        )

def _report_iteration(reporter, step_id, n, it, res_norm, res, fd, du):
    res_max = float(np.max(np.abs(res))) if res.size else 0.0
    reporter(
        IterationReport(
            step_id=step_id, inc=n + 1, attempt=1, it=it,
            residual_norm=res_norm, residual_max=res_max,
            residual_dof=None, residual_node=None, residual_component_label="FORCE",
            correction_norm=float(np.linalg.norm(du)),
            correction_max=float(np.max(np.abs(du))) if du.size else 0.0,
            converged_force=False, converged_moment=False, note=None,
        )
    )
