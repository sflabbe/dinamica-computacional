from __future__ import annotations

import numpy as np

from dinamica_computacional.core.model import Model


def solve_static_newton(model: Model, max_iter: int = 60, tol: float = 1e-10) -> np.ndarray:
    nd = model.ndof()
    fd = model.free_dofs()

    u = np.zeros(nd)
    u_free = u[fd].copy()

    for _ in range(max_iter):
        u_trial = u.copy()
        u_trial[fd] = u_free
        model.update_column_yields(u_trial)
        K, Rint, _ = model.assemble(u_trial, u)
        res = model.load_const[fd] - Rint
        if np.linalg.norm(res) < tol * max(1.0, np.linalg.norm(model.load_const[fd])):
            u = u_trial.copy()
            model.commit()
            return u
        du = np.linalg.solve(K + 1e-14 * np.eye(fd.size), res)
        u_free += du

    raise RuntimeError("Static Newton did not converge")
