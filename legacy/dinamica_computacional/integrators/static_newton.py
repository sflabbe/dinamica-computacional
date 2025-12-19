from __future__ import annotations

import numpy as np

from dinamica_computacional.core.model import Model


def solve_static_newton(
    model: Model,
    max_iter: int = 60,
    tol: float = 1e-10,
    load_steps: int = 10,
    min_load_step: float = 1e-3,
) -> np.ndarray:
    nd = model.ndof()
    fd = model.free_dofs()

    u = np.zeros(nd)
    u_free = u[fd].copy()
    total_load = model.load_const[fd]

    load_factor = 0.0
    step_size = 1.0 / max(load_steps, 1)
    min_step = max(min_load_step, 1e-8)

    while load_factor < 1.0 - 1e-12:
        target = min(load_factor + step_size, 1.0)
        load = total_load * target
        for _ in range(max_iter):
            u_trial = u.copy()
            u_trial[fd] = u_free
            model.update_column_yields(u_trial)
            K, Rint, _ = model.assemble(u_trial, u)
            res = load - Rint
            if np.linalg.norm(res) < tol * max(1.0, np.linalg.norm(load)):
                u = u_trial.copy()
                model.commit()
                load_factor = target
                break
            du = np.linalg.solve(K + 1e-14 * np.eye(fd.size), res)
            u_free += du
        else:
            if step_size <= min_step:
                raise RuntimeError(
                    f"Static Newton did not converge (load factor {target:.4f}, step {step_size:.2e})"
                )
            step_size *= 0.5
            u_free = u[fd].copy()

    return u
