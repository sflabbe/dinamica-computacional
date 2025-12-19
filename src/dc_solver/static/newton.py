"""Newton-Raphson static solver."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from dc_solver.fem.model import Model
from dc_solver.reporting import IterationReport


def solve_static_newton(
    model: Model,
    load: np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-10,
    reporter: Optional[Callable[[object], None]] = None,
    step_id: int = 1,
    inc: int = 1,
    attempt: int = 1,
    stats: Optional[dict] = None,
) -> np.ndarray:
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size

    u = np.zeros(nd)
    u_free = u[fd].copy()
    for it in range(1, max_iter + 1):
        u_trial = u.copy()
        u_trial[fd] = u_free
        model.update_column_yields(u_trial)
        K, Rint, _ = model.assemble(u_trial, u)
        res = load[fd] - Rint
        res_norm = float(np.linalg.norm(res))
        res_ref = max(1.0, float(np.linalg.norm(load[fd])))
        res_max = float(np.max(np.abs(res))) if res.size else 0.0
        res_idx = int(np.argmax(np.abs(res))) if res.size else None
        res_dof = int(fd[res_idx]) if res_idx is not None else None
        if res_norm < tol * res_ref:
            if reporter is not None:
                reporter(
                    IterationReport(
                        step_id=step_id,
                        inc=inc,
                        attempt=attempt,
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
            u = u_trial.copy()
            model.commit()
            if stats is not None:
                stats["iters"] = it
            return u
        du = np.linalg.solve(K + 1e-14 * np.eye(nf), res)
        if reporter is not None:
            reporter(
                IterationReport(
                    step_id=step_id,
                    inc=inc,
                    attempt=attempt,
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
    raise RuntimeError("No converge el paso estático.")
