"""Newmark-beta integration with Newton iterations."""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IncrementEnd, IncrementStart


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
    reporter: Optional[Callable[[object], None]] = None,
    step_id: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Constant-average-acceleration Newmark method (implicit) with Newton iterations.

    Output keys aim to match hht_alpha_newton for downstream plotting.
    """
    solve_start = time.perf_counter()

    model.reset_state()
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size

    # Static gravity equilibration unless u0/v0 provided
    if u0 is None and v0 is None:
        u = np.zeros(nd)
        u_free = u[fd].copy()
        for _ in range(60):
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
            raise RuntimeError("No converge el paso estático de gravedad.")
        u_n = u.copy()
        v_n = np.zeros(nd)
    else:
        u_n = np.zeros(nd) if u0 is None else u0.copy()
        v_n = np.zeros(nd) if v0 is None else v0.copy()

    a_n = np.zeros(nd)

    M = model.mass_diag
    C = model.C_diag
    dt = float(t[1] - t[0])

    # influence vector (same assumption as HHT)
    r = np.zeros(nd)
    r[np.where(M > 0.0)[0]] = 1.0

    # Newmark constants
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)

    if verbose:
        print(f"[NEWMARK] steps={t.size - 1} dt={dt:.6f}s beta={beta:.3f} gamma={gamma:.3f}")

    # Initialize acceleration if u0/v0 provided
    if u0 is not None or v0 is not None:
        model.update_column_yields(u_n)
        Rint0 = model.internal_force(u_n)
        p0 = model.load_const.copy()
        if load_hist is not None and load_hist.shape[0] > 0:
            p0 = p0 + load_hist[0]
        p0 = p0 - M * r * ag[0]
        mass_mask = M > 0.0
        a_n[mass_mask] = (p0[mass_mask] - Rint0[mass_mask] - C[mass_mask] * v_n[mass_mask]) / M[mass_mask]

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

    if load_hist is None:
        load_hist = np.zeros((t.size, nd))
    p_const = model.load_const.copy()

    snapshot_idx: Optional[int] = None
    snapshot_reached = False
    snapshot_t = float("nan")
    snapshot_drift = float("nan")

    iters_total = 0

    for n in range(t.size - 1):
        model.update_column_yields(u_n)

        p_np1 = p_const + load_hist[n + 1] - M * r * ag[n + 1]

        # predictors
        u_pred = u_n + dt * v_n + 0.5 * dt * dt * (1.0 - 2.0 * beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy()

        if reporter is not None:
            reporter(IncrementStart(step_id=step_id, inc=n + 1, attempt=1, dt=float(dt)))

        converged = False
        last_inf = {"hinges": []}

        for it in range(max_iter):
            u_trial = u_comm_step.copy()
            u_trial[fd] = u_free

            K_tan, Rint, inf = model.assemble(u_trial, u_comm_step)
            last_inf = inf

            a_trial = a0 * (u_trial - u_pred)
            v_trial = v_pred + gamma * dt * a_trial

            res = p_np1[fd] - Rint - np.diag(M[fd] * a0) @ (u_trial[fd] - u_pred[fd]) - C[fd] * v_trial[fd]

            # note: res is already on free dofs, so norm on res
            if np.linalg.norm(res) < tol * max(1.0, np.linalg.norm(p_np1[fd])):
                # accept
                u_n = u_trial
                v_n = v_trial
                a_n = a_trial
                model.commit()
                iters[n] = it
                iters_total += it
                hinge_hist.append(inf.get("hinges", []))
                converged = True
                if reporter is not None:
                    reporter(
                        IncrementEnd(
                            step_id=step_id,
                            inc=n + 1,
                            attempt=1,
                            converged=True,
                            n_equil_iters=it,
                            n_severe_iters=0,
                            dt_completed=float(dt),
                            step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
                            step_time_completed=float(t[n + 1]),
                            total_time_completed=float(t[n + 1]),
                        )
                    )
                break

            K_eff = K_tan + np.diag(M[fd] * a0) + np.diag(C[fd] * a1)

            try:
                du = np.linalg.solve(K_eff + 1e-14 * np.eye(nf), res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K_eff + 1e-14 * np.eye(nf), res, rcond=None)[0]
            u_free += du

        if not converged:
            # still store trial results to avoid crashing downstream
            u_n = u_trial
            v_n = v_pred
            a_n = a0 * (u_trial - u_pred)
            model.commit()
            hinge_hist.append(last_inf.get("hinges", []))
            if reporter is not None:
                reporter(
                    IncrementEnd(
                        step_id=step_id,
                        inc=n + 1,
                        attempt=1,
                        converged=False,
                        n_equil_iters=max_iter,
                        n_severe_iters=0,
                        dt_completed=float(dt),
                        step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
                        step_time_completed=float(t[n + 1]),
                        total_time_completed=float(t[n + 1]),
                    )
                )
            # keep going

        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        drift[n + 1] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
        Vb[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)

        if (not snapshot_reached) and abs(drift[n + 1]) >= drift_snapshot:
            snapshot_reached = True
            snapshot_idx = n + 1
            snapshot_t = float(t[n + 1])
            snapshot_drift = float(drift[n + 1])

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
        "beta": beta,
        "gamma": gamma,
        "n_steps": int(t.size - 1),
        "iters_total": int(iters_total),
        "solve_time_s": float(solve_time_s),
        "snapshot_idx": int(snapshot_idx),
        "snapshot_t": float(snapshot_t),
        "snapshot_drift": float(snapshot_drift),
        "snapshot_limit": float(drift_snapshot),
        "snapshot_reached": bool(snapshot_reached),
    }
