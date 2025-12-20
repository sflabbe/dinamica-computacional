"""FLOPs estimation utilities for performance reporting."""

from __future__ import annotations

from typing import Dict, Any, Optional


def estimate_flops_dynamics(
    *,
    ndof: int,
    n_steps: int,
    integrator: str = "explicit",
    avg_iterations: float = 1.0,
    n_elements: Optional[int] = None,
) -> float:
    """Estimate FLOPs for a dynamics analysis.

    This is a rough order-of-magnitude estimate based on the dominant operations:
    - Element force/stiffness assembly: O(n_elements * dof_per_elem^3)
    - Matrix-vector products: O(ndof^2) for dense, O(ndof) for sparse
    - Solving linear systems: O(ndof^3) for dense direct, O(ndof) for explicit

    For frame elements with 6 DOF/element and sparse operations, we use simplified
    estimates suitable for reporting purposes.

    Parameters
    ----------
    ndof : int
        Total number of degrees of freedom
    n_steps : int
        Number of time steps
    integrator : str
        Integrator type: 'explicit', 'implicit', 'newmark', 'hht'
    avg_iterations : float, default=1.0
        Average number of iterations per step (for implicit methods)
    n_elements : Optional[int]
        Number of elements (if known, improves estimate)

    Returns
    -------
    float
        Estimated floating-point operations count

    Notes
    -----
    These are VERY rough estimates for reporting purposes only:
    - Explicit: ~100*ndof FLOPs/step (force eval + explicit update)
    - Implicit: ~1000*ndof FLOPs/step/iteration (includes Jacobian, solve, etc.)

    For more accurate profiling, use actual FLOP counters or profilers.

    Examples
    --------
    >>> estimate_flops_dynamics(ndof=300, n_steps=10000, integrator='explicit')
    300000000.0
    >>> estimate_flops_dynamics(ndof=300, n_steps=1000, integrator='implicit', avg_iterations=5)
    1500000000.0
    """
    if ndof <= 0 or n_steps <= 0:
        return 0.0

    # Rough per-step FLOPs estimates
    if integrator.lower() in ("explicit",):
        # Element assembly + vector updates
        # ~50 FLOPs/DOF for force eval, ~50 FLOPs/DOF for state updates
        flops_per_step = 100.0 * ndof
    elif integrator.lower() in ("implicit", "newmark", "hht", "hht_alpha"):
        # Each iteration: element assembly (~100 FLOP/DOF), Jacobian assembly (~500 FLOP/DOF),
        # sparse solve (very rough ~300 FLOP/DOF), updates (~100 FLOP/DOF)
        # Total ~1000 FLOP/DOF per iteration
        flops_per_iter = 1000.0 * ndof
        flops_per_step = flops_per_iter * max(1.0, avg_iterations)
    else:
        # Default fallback
        flops_per_step = 200.0 * ndof

    total_flops = flops_per_step * n_steps
    return float(total_flops)


def compute_gflops_rate(flops: float, wall_seconds: float) -> float:
    """Compute GFLOP/s rate.

    Parameters
    ----------
    flops : float
        Total floating-point operations
    wall_seconds : float
        Wall-clock time in seconds

    Returns
    -------
    float
        GFLOP/s rate (flops / 1e9 / wall_seconds)
    """
    if wall_seconds <= 0.0:
        return 0.0
    return flops / 1e9 / wall_seconds


def build_flops_report(
    *,
    ndof: int,
    n_steps: int,
    integrator: str,
    avg_iterations: float = 1.0,
    wall_seconds: float = 0.0,
    n_elements: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a FLOPs report dictionary.

    Parameters
    ----------
    ndof : int
        Number of DOFs
    n_steps : int
        Number of steps
    integrator : str
        Integrator name
    avg_iterations : float, default=1.0
        Average iterations per step
    wall_seconds : float, default=0.0
        Total wall time
    n_elements : Optional[int]
        Number of elements

    Returns
    -------
    Dict[str, Any]
        Report with keys: flops_est, gflops_rate, ndof, n_steps, integrator
    """
    flops_est = estimate_flops_dynamics(
        ndof=ndof,
        n_steps=n_steps,
        integrator=integrator,
        avg_iterations=avg_iterations,
        n_elements=n_elements,
    )
    gflops_rate = compute_gflops_rate(flops_est, wall_seconds)

    return {
        "flops_est": flops_est,
        "gflops_rate": gflops_rate,
        "ndof": ndof,
        "n_steps": n_steps,
        "integrator": integrator,
        "avg_iterations": avg_iterations,
        "wall_seconds": wall_seconds,
    }
