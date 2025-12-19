"""HHT-alpha integration with Newton iterations."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Callable

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IncrementEnd, IncrementStart, IterationReport


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
    reporter: Optional[Callable[[object], None]] = None,
    step_id: int = 1,
) -> Dict[str, np.ndarray]:
    if not (-1.0 / 3.0 - 1e-12 <= alpha <= 1e-12):
        raise ValueError("HHT-alpha requires alpha in [-1/3, 0].")

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
    gamma = 0.5 - alpha
    beta = 0.25 * (1.0 - alpha) ** 2
    rho_inf = (1.0 + alpha) / (1.0 - alpha)
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)

    if verbose:
        print(
            f"[HHT] steps={t.size - 1} dt={dt:.6f}s alpha={alpha:.3f} "
            f"beta={beta:.4f} gamma={gamma:.4f}"
        )

    r = np.zeros(nd)
    r[np.where(M > 0.0)[0]] = 1.0

    u_hist = np.zeros((t.size, nd))
    v_hist = np.zeros((t.size, nd))
    a_hist = np.zeros((t.size, nd))
    drift = np.zeros(t.size)
    Vb = np.zeros(t.size)
    iters = np.zeros(t.size - 1, dtype=int)

    hinge_hist = []
    snapshot_idx: Optional[int] = None

    ux2 = model.nodes[drift_nodes[0]].dof_u[0]
    ux3 = model.nodes[drift_nodes[1]].dof_u[0]

    if u0 is not None or v0 is not None:
        model.update_column_yields(u_n)
        Rint0 = model.internal_force(u_n)
        p0 = model.load_const.copy()
        if load_hist is not None and load_hist.shape[0] > 0:
            p0 = p0 + load_hist[0]
        p0 = p0 - M * r * ag[0]
        mass_mask = M > 0.0
        a_n[mass_mask] = (p0[mass_mask] - Rint0[mass_mask] - C[mass_mask] * v_n[mass_mask]) / M[mass_mask]

    u_hist[0] = u_n
    v_hist[0] = v_n
    a_hist[0] = a_n
    drift[0] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
    Vb[0] = model.base_shear(u_n, base_nodes=base_nodes)

    p_const = model.load_const.copy()
    if load_hist is None:
        load_hist = np.zeros((t.size, nd))

    for n in range(t.size - 1):
        model.update_column_yields(u_n)

        p_n = p_const + load_hist[n] - M * r * ag[n]
        p_np1 = p_const + load_hist[n + 1] - M * r * ag[n + 1]
        p_alpha = (1.0 + alpha) * p_np1 - alpha * p_n

        _, Rint_n, _ = model.assemble(u_n, u_n)

        u_pred = u_n + dt * v_n + dt * dt * (0.5 - beta) * a_n
        v_pred = v_n + dt * (1.0 - gamma) * a_n

        u_free = u_pred[fd].copy()
        u_comm_step = u_n.copy()

        if reporter is not None:
            reporter(
                IncrementStart(
                    step_id=step_id,
                    inc=n + 1,
                    attempt=1,
                    dt=dt,
                    step_time=float(t[n]),
                    total_time=float(t[n]),
                    is_cutback_attempt=False,
                )
            )

        if verbose:
            print(f"[HHT] inc {n + 1}/{t.size - 1} t={t[n + 1]:.3f}s")

        for it in range(1, max_iter + 1):
            u_trial = u_comm_step.copy()
            u_trial[fd] = u_free

            K_tan, Rint, inf = model.assemble(u_trial, u_comm_step)

            a_trial = a0 * (u_trial - u_pred)
            v_trial = v_pred + (gamma * dt) * a_trial

            res = p_alpha[fd] - (
                (1.0 + alpha) * Rint - alpha * Rint_n + C[fd] * v_trial[fd] + M[fd] * a_trial[fd]
            )

            res_norm = float(np.linalg.norm(res))
            res_max = float(np.max(np.abs(res))) if res.size else 0.0
            res_idx = int(np.argmax(np.abs(res))) if res.size else None
            res_dof = int(fd[res_idx]) if res_idx is not None else None
            scale = 1.0 + np.linalg.norm(p_alpha[fd])
            if res_norm <= tol * scale:
                if reporter is not None:
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
                            correction_norm=0.0,
                            correction_max=0.0,
                            converged_force=True,
                            converged_moment=True,
                            note=None,
                        )
                    )
                if verbose:
                    print(f"[HHT]   converged in {it} iters (|res|={res_norm:.3e})")
                u_n = u_trial
                v_n = v_trial
                a_n = a_trial
                model.commit()
                iters[n] = it
                hinge_hist.append(inf["hinges"])
                if reporter is not None:
                    reporter(
                        IncrementEnd(
                            step_id=step_id,
                            inc=n + 1,
                            attempt=1,
                            converged=True,
                            n_equil_iters=it,
                            n_severe_iters=0,
                            dt_completed=dt,
                            step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
                            step_time_completed=float(t[n + 1]),
                            total_time_completed=float(t[n + 1]),
                        )
                    )
                break

            K_eff = (
                (1.0 + alpha) * K_tan
                + np.diag(M[fd] * a0)
                + np.diag(C[fd] * a1)
            )
            try:
                du = np.linalg.solve(K_eff + 1e-14 * np.eye(nf), res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(K_eff + 1e-14 * np.eye(nf), res, rcond=None)[0]
            if reporter is not None:
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
            u_free += du

        else:
            if reporter is not None:
                reporter(
                    IncrementEnd(
                        step_id=step_id,
                        inc=n + 1,
                        attempt=1,
                        converged=False,
                        n_equil_iters=max_iter,
                        n_severe_iters=0,
                        dt_completed=dt,
                        step_fraction=float((t[n + 1] / t[-1]) if t[-1] != 0 else 1.0),
                        step_time_completed=float(t[n + 1]),
                        total_time_completed=float(t[n + 1]),
                    )
                    )
            if verbose:
                print(f"[HHT]   failed to converge after {max_iter} iters")
            raise RuntimeError(f"No converge en paso {n+1} / t={t[n+1]:.3f}s (HHT-alpha).")

        u_hist[n + 1] = u_n
        v_hist[n + 1] = v_n
        a_hist[n + 1] = a_n
        drift[n + 1] = 0.5 * (u_n[ux2] + u_n[ux3]) / drift_height
        Vb[n + 1] = model.base_shear(u_n, base_nodes=base_nodes)
        if snapshot_idx is None and abs(drift[n + 1]) >= drift_snapshot:
            snapshot_idx = n + 1

        if abs(drift[n + 1]) >= drift_limit:
            if verbose:
                print(
                    f"COLLAPSE by drift >= {100*drift_limit:.1f}% at t={t[n+1]:.3f}s"
                )
            u_hist = u_hist[: n + 2]
            v_hist = v_hist[: n + 2]
            a_hist = a_hist[: n + 2]
            drift = drift[: n + 2]
            Vb = Vb[: n + 2]
            t = t[: n + 2]
            ag = ag[: n + 2]
            iters = iters[: n + 1]
            break

    n_steps = iters.size
    iters_total = int(np.sum(iters)) if iters.size else 0
    solve_time_s = time.perf_counter() - solve_start
    if snapshot_idx is None:
        snapshot_idx = int(np.argmax(np.abs(drift))) if drift.size else -1
    snapshot_drift = (
        float(drift[int(snapshot_idx)])
        if (snapshot_idx >= 0 and snapshot_idx < drift.size)
        else float("nan")
    )
    snapshot_t = (
        float(t[int(snapshot_idx)])
        if (snapshot_idx >= 0 and snapshot_idx < t.size)
        else float("nan")
    )
    snapshot_reached = bool(abs(snapshot_drift) >= drift_snapshot) if np.isfinite(snapshot_drift) else False

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
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "rho_inf": rho_inf,
        "n_steps": n_steps,
        "iters_total": iters_total,
        "solve_time_s": solve_time_s,
        "snapshot_idx": int(snapshot_idx),
        "snapshot_t": snapshot_t,
        "snapshot_drift": snapshot_drift,
        "snapshot_limit": drift_snapshot,
        "snapshot_reached": snapshot_reached,
    }
