"""Newton-Raphson static solver."""

from __future__ import annotations

import numpy as np

from dc_solver.fem.model import Model


def solve_static_newton(
    model: Model,
    load: np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-10,
) -> np.ndarray:
    nd = model.ndof()
    fd = model.free_dofs()
    nf = fd.size

    u = np.zeros(nd)
    u_free = u[fd].copy()
    for _ in range(max_iter):
        u_trial = u.copy()
        u_trial[fd] = u_free
        model.update_column_yields(u_trial)
        K, Rint, _ = model.assemble(u_trial, u)
        res = load[fd] - Rint
        if np.linalg.norm(res) < tol * max(1.0, np.linalg.norm(load[fd])):
            u = u_trial.copy()
            model.commit()
            return u
        du = np.linalg.solve(K + 1e-14 * np.eye(nf), res)
        u_free += du
    raise RuntimeError("No converge el paso estático.")
