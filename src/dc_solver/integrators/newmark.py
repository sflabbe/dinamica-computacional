"""Newmark-beta integration with Newton iterations."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from dc_solver.fem.model import Model


def newmark_beta_newton(
    model: Model,
    t: np.ndarray,
    ag: np.ndarray,
    drift_height: float,
    base_nodes: Tuple[int, int],
    drift_nodes: Tuple[int, int],
    beta: float = 0.25,
    gamma: float = 0.5,
    max_iter: int = 30,
    tol: float = 1e-6,
    u0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    load_hist: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    solve_start = time.perf_counter()

    model.reset_state()
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size

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
            du = np.linalg.solve(K + 1e-14 * np.eye(nf), res)
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
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)

    r = np.zeros(nd)
    r[np.where(M > 0.0)[0]] = 1.0

    u_hist = np.zeros((t.size, nd))
    v_hist = np.zeros((t.size, nd))
    a_hist = np.zeros((t.size, nd))

    ux2 = model.nodes[drift_nodes[0]].dof_u[0]
    ux3 = model.nodes[drift_nodes[1]].dof_u[0]
    drift = np.zeros(t.size)
    Vb = np.zeros(t.size)
    iters = np.zeros(t.size - 1, dtype=int)

    if load_hist is None:
        load_hist = np.zeros((t.size, nd))

    if u0 is not None or v0 is not None:
        model.update_column_yields(u_n)
        Rint0 = model.internal_force(u_n)
        p0 = model.load_const.copy() + load_hist[0] - M * r * ag[0]
        mass_mask = M > 0.0
        a_n[mass_mask] = (p0[mass_mask] - Rint0[mass_mask] - C[mass_mask] * v_n[mass_mask]) / M[mass_mask]

    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift[0] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
    Vb[0] = model.base_shear(u_n, base_nodes=base_nodes)

    p_const = model.load_const.copy()

    for n in range(t.size - 1):
        model.update_column_yields(u_n)

        p_np1 = p_const + load_hist[n + 1] - M * r * ag[n + 1]

        u_pred = u_n + dt * v_n + dt * dt * (0.5 - beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy()

        for it in range(max_iter):
            u_trial = u_comm_step.copy()
            u_trial[fd] = u_free

            K_tan, Rint, _ = model.assemble(u_trial, u_comm_step)

            a_trial = a0 * (u_trial - u_pred)
            v_trial = v_pred + (gamma * dt) * a_trial

            res = p_np1[fd] - (Rint + C[fd] * v_trial[fd] + M[fd] * a_trial[fd])

            scale = 1.0 + np.linalg.norm(p_np1[fd])
            if np.linalg.norm(res) <= tol * scale:
                u_n = u_trial
                v_n = v_trial
                a_n = a_trial
                model.commit()
                iters[n] = it + 1
                break

            K_eff = (
                K_tan
                + np.diag(M[fd] * a0)
                + np.diag(C[fd] * a1)
            )
            du = np.linalg.solve(K_eff + 1e-14 * np.eye(nf), res)
            u_free += du

        else:
            raise RuntimeError(f"No converge en paso {n+1} / t={t[n+1]:.3f}s (Newmark).")

        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        drift[n + 1] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
        Vb[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)

    solve_time_s = time.perf_counter() - solve_start

    return {
        "t": t,
        "ag": ag,
        "u": u_hist,
        "v": v_hist,
        "a": a_hist,
        "drift": drift,
        "Vb": Vb,
        "iters": iters,
        "dt": dt,
        "beta": beta,
        "gamma": gamma,
        "n_steps": iters.size,
        "solve_time_s": solve_time_s,
    }
