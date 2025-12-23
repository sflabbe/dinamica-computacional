"""Gravity-only static solve utilities.

Provides a robust Newton-Raphson solver intended for initializing models under gravity
(State 1). Designed to work with the educational dc_solver Model API.

Key points (important for hinge models):
  - Load is ramped over n_load_steps.
  - Within each load step, Newton iterations use a fixed reference state u_ref
    (the converged state from the previous load step). This avoids treating Newton
    iterations as physical increments (which breaks plastic hinges / history variables).
  - commit() is called only after convergence of each load step.
  - Returns roof/base metrics (ux_roof, uy_roof, drift) and base shear Vb if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Numerical settings
EPS_REG = 1e-14


@dataclass
class GravityResult:
    u: np.ndarray
    converged: bool
    iters_total: int
    res_norm: float
    load_steps: int
    # Convenience metrics
    ux_roof: float
    uy_roof: float
    ux_base: float
    uy_base: float
    drift: float
    Vb: float
    roof_nodes: Tuple[int, ...]
    base_nodes: Tuple[int, ...]


def _roof_base_node_sets(model) -> Tuple[Tuple[int, ...], Tuple[int, ...], float]:
    ys = np.array([nd.y for nd in model.nodes], dtype=float)
    ymax = float(np.max(ys))
    ymin = float(np.min(ys))
    roof = tuple(i for i, nd in enumerate(model.nodes) if abs(nd.y - ymax) < 1e-9)
    base = tuple(i for i, nd in enumerate(model.nodes) if abs(nd.y - ymin) < 1e-9)
    H = (ymax - ymin) if ymax > ymin else 1.0
    # ensure at least one index
    if len(roof) == 0:
        roof = (int(np.argmax(ys)),)
    if len(base) == 0:
        base = (int(np.argmin(ys)),)
    return roof, base, H


def _mean_dof(u: np.ndarray, model, node_ids: Tuple[int, ...], dof_local: int) -> float:
    vals = []
    for i in node_ids:
        dof = model.nodes[i].dof_u[dof_local]
        vals.append(u[dof])
    return float(np.mean(vals)) if vals else 0.0


def solve_gravity_only(
    model,
    *,
    tol: float = 1e-10,
    max_iter: int = 80,
    n_load_steps: int = 10,
    allow_substepping: bool = True,
    min_load_step: float = 1.0 / 2048.0,
    max_total_steps: int = 20000,
    line_search: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Solve static equilibrium under model.load_const using Newton-Raphson.

    Notes
    -----
    - Load is ramped from 0 → 1.
    - History-dependent constitutive models (plastic hinges / fiber hinges) must
      NOT be committed within Newton iterations; only commit after convergence.
    - If `allow_substepping=True`, the solver will automatically reduce the load
      increment when a load step fails to converge (common for fiber hinges).

    Returns a dict compatible with prior callers, and includes additional metrics.
    """
    model.reset_state()

    nd = model.ndof()
    fd = model.free_dofs()
    nf = int(fd.size)

    u = np.zeros(nd, dtype=float)
    u_free = u[fd].copy()

    # Full external load (gravity) is assumed stored here.
    P_full = np.array(model.load_const, dtype=float).copy()

    iters_total = 0
    res_norm = np.nan

    # Adaptive load stepping
    ds0 = 1.0 / float(max(1, int(n_load_steps)))
    ds = ds0
    s = 0.0
    step_count = 0
    n_halvings = 0

    while s < 1.0 - 1e-15:
        if step_count >= int(max_total_steps):
            raise RuntimeError(
                f"Gravity-only exceeded max_total_steps={max_total_steps} "
                f"(last scale={s:.6f}, ds={ds:.6e}, |res|={res_norm:.3e})."
            )

        s_target = min(1.0, s + ds)
        P_step = s_target * P_full

        # Freeze reference at start of attempted load step (last converged state)
        u_ref = u.copy()

        if verbose:
            print(f"[gravity] try step: scale {s:.6f} -> {s_target:.6f} (ds={ds:.6e})")

        converged = False
        # Newton loop
        for it in range(int(max_iter)):
            iters_total += 1

            u_trial = u_ref.copy()
            u_trial[fd] = u_free

            if hasattr(model, 'update_column_yields'):
                model.update_column_yields(u_trial)

            # Assemble at (u_trial, u_ref) to keep history consistent during Newton
            K, Rint_f, _ = model.assemble(u_trial, u_ref)

            res = P_step[fd] - Rint_f
            res_norm = float(np.linalg.norm(res))

            ref = max(1.0, float(np.linalg.norm(P_step[fd])))
            if res_norm <= tol * ref:
                model.commit()
                u = u_trial
                u_free = u[fd].copy()
                converged = True
                break

            mat = K + EPS_REG * np.eye(nf)
            try:
                du = np.linalg.solve(mat, res)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(mat, res, rcond=None)[0]

            if line_search:
                alpha = 1.0
                u_free_try = u_free + alpha * du
                res0 = res_norm
                for _ls in range(12):
                    u_trial_ls = u_ref.copy()
                    u_trial_ls[fd] = u_free_try
                    if hasattr(model, 'update_column_yields'):
                        model.update_column_yields(u_trial_ls)
                    _, Rint_f_ls, _ = model.assemble(u_trial_ls, u_ref)
                    res_ls = P_step[fd] - Rint_f_ls
                    res_ls_norm = float(np.linalg.norm(res_ls))
                    if res_ls_norm <= (1.0 - 1e-4 * alpha) * res0:
                        u_free = u_free_try
                        break
                    alpha *= 0.5
                    u_free_try = u_free + alpha * du
                else:
                    u_free = u_free + 0.25 * du
            else:
                u_free = u_free + du

        if converged:
            s = s_target
            step_count += 1
            # Optional: gently grow ds back towards ds0 after successful substeps
            if allow_substepping and ds < ds0:
                ds = min(ds0, ds * 1.25)
            continue

        # Not converged: handle substepping
        if not allow_substepping:
            raise RuntimeError(
                f"Gravity-only Newton did not converge in {max_iter} iterations "
                f"at load scale={s_target:.6f} (|res|={res_norm:.3e})."
            )

        if ds <= float(min_load_step) + 1e-18:
            raise RuntimeError(
                f"Gravity-only Newton did not converge even after substepping "
                f"(min_load_step={min_load_step:.3e}). Last attempted scale={s_target:.6f}, "
                f"ds={ds:.3e}, |res|={res_norm:.3e}. "
                f"Tip: try --line-search or increase --gravity-max-iter / --gravity-steps."
            )

        ds *= 0.5
        n_halvings += 1
        # Reset trial u_free to last converged u (avoid accumulating failed steps)
        u_free = u[fd].copy()
        if verbose:
            print(
                f"[gravity] non-convergence at scale={s_target:.6f}; "
                f"substepping: new ds={ds:.6e} (halvings={n_halvings}, |res|={res_norm:.3e})"
            )

    # Final metrics
    roof_nodes, base_nodes, H = _roof_base_node_sets(model)
    ux_roof = _mean_dof(u, model, roof_nodes, 0)
    uy_roof = _mean_dof(u, model, roof_nodes, 1)
    ux_base = _mean_dof(u, model, base_nodes, 0)
    uy_base = _mean_dof(u, model, base_nodes, 1)
    drift = (ux_roof - ux_base) / H

    # Base shear (horizontal). If model provides helper, use it; else NaN.
    Vb = float("nan")
    if hasattr(model, "base_shear"):
        # model.base_shear expects a tuple of two base node indices in your repo.
        if len(base_nodes) >= 2:
            bn = (int(base_nodes[0]), int(base_nodes[1]))
        else:
            bn = (int(base_nodes[0]), int(base_nodes[0]))
        try:
            Vb = float(model.base_shear(u, base_nodes=bn))
        except Exception:
            Vb = float("nan")

    out: Dict[str, Any] = {
        "u": u,
        "converged": True,
        "iters_total": int(iters_total),
        "res_norm": float(res_norm),
        "load_steps": int(n_load_steps),
        # metrics
        "ux_roof": float(ux_roof),
        "uy_roof": float(uy_roof),
        "ux_base": float(ux_base),
        "uy_base": float(uy_base),
        "drift": float(drift),
        "Vb": float(Vb),
        "roof_nodes": roof_nodes,
        "base_nodes": base_nodes,
    }
    return out
