from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ._numba import njit, USE_NUMBA


@njit(cache=True)
def _inv2(W: np.ndarray) -> np.ndarray:
    det = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]
    if abs(det) < 1e-18:
        # fall back to identity-like (should not happen for PD matrices)
        out = np.zeros((2, 2), dtype=np.float64)
        out[0, 0] = 1.0
        out[1, 1] = 1.0
        return out
    inv_det = 1.0 / det
    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = W[1, 1] * inv_det
    out[0, 1] = -W[0, 1] * inv_det
    out[1, 0] = -W[1, 0] * inv_det
    out[1, 1] = W[0, 0] * inv_det
    return out


@njit(cache=True)
def _obj_quadratic(x: np.ndarray, x0: np.ndarray, W: np.ndarray) -> float:
    dx0 = x[0] - x0[0]
    dx1 = x[1] - x0[1]
    # 0.5 * d^T W d
    return 0.5 * (dx0 * (W[0, 0] * dx0 + W[0, 1] * dx1) + dx1 * (W[1, 0] * dx0 + W[1, 1] * dx1))


@njit(cache=True)
def _is_feasible(x: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float) -> bool:
    m = A.shape[0]
    for i in range(m):
        if A[i, 0] * x[0] + A[i, 1] * x[1] - b[i] > tol + 1e-12:
            return False
    return True


@njit(cache=True)
def _project_one(x0: np.ndarray, a0: float, a1: float, bi: float, W: np.ndarray, Winv: np.ndarray) -> Tuple[np.ndarray, float]:
    # Solve scalar KKT:
    # x = x0 - Winv a * lam ; lam = (a x0 - b) / (a Winv a^T)
    denom = a0 * (Winv[0, 0] * a0 + Winv[0, 1] * a1) + a1 * (Winv[1, 0] * a0 + Winv[1, 1] * a1)
    if abs(denom) < 1e-18:
        x = x0.copy()
        return x, 0.0
    rhs = a0 * x0[0] + a1 * x0[1] - bi
    lam = rhs / denom
    x = np.empty(2, dtype=np.float64)
    x[0] = x0[0] - (Winv[0, 0] * a0 + Winv[0, 1] * a1) * lam
    x[1] = x0[1] - (Winv[1, 0] * a0 + Winv[1, 1] * a1) * lam
    return x, lam


@njit(cache=True)
def _solve_2x2(S: np.ndarray, rhs: np.ndarray) -> Tuple[bool, np.ndarray]:
    det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
    if abs(det) < 1e-18:
        return False, np.zeros(2, dtype=np.float64)
    inv_det = 1.0 / det
    lam = np.empty(2, dtype=np.float64)
    lam[0] = (rhs[0] * S[1, 1] - rhs[1] * S[0, 1]) * inv_det
    lam[1] = (-rhs[0] * S[1, 0] + rhs[1] * S[0, 0]) * inv_det
    return True, lam


@njit(cache=True)
def _project_two(x0: np.ndarray, Ai: np.ndarray, bi: np.ndarray, W: np.ndarray, Winv: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
    # Solve KKT for 2 actives:
    # S lam = rhs ; x = x0 - Winv A^T lam
    # where S = A Winv A^T (2x2), rhs = A x0 - b (2,)
    S = np.empty((2, 2), dtype=np.float64)
    # Build S explicitly
    # row p: ap = Ai[p,:]
    for p in range(2):
        ap0 = Ai[p, 0]
        ap1 = Ai[p, 1]
        for q in range(2):
            aq0 = Ai[q, 0]
            aq1 = Ai[q, 1]
            # ap * Winv * aq^T
            S[p, q] = ap0 * (Winv[0, 0] * aq0 + Winv[0, 1] * aq1) + ap1 * (Winv[1, 0] * aq0 + Winv[1, 1] * aq1)

    rhs = np.empty(2, dtype=np.float64)
    rhs[0] = Ai[0, 0] * x0[0] + Ai[0, 1] * x0[1] - bi[0]
    rhs[1] = Ai[1, 0] * x0[0] + Ai[1, 1] * x0[1] - bi[1]

    ok, lam = _solve_2x2(S, rhs)
    if not ok:
        return False, x0.copy(), np.zeros(2, dtype=np.float64)

    x = np.empty(2, dtype=np.float64)
    # Winv A^T lam:
    at0 = Ai[0, 0] * lam[0] + Ai[1, 0] * lam[1]
    at1 = Ai[0, 1] * lam[0] + Ai[1, 1] * lam[1]
    x[0] = x0[0] - (Winv[0, 0] * at0 + Winv[0, 1] * at1)
    x[1] = x0[1] - (Winv[1, 0] * at0 + Winv[1, 1] * at1)
    return True, x, lam


@njit(cache=True)
def _project_onto_polytope_2d_kernel(
    x0: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, int, int, int, float, float, float]:
    # returns: x, n_active, i0, i1, lam0, lam1, obj
    x0 = x0.reshape(2)
    m = A.shape[0]

    # inside check
    inside = True
    for i in range(m):
        if A[i, 0] * x0[0] + A[i, 1] * x0[1] - b[i] > tol:
            inside = False
            break
    if inside:
        return x0.copy(), 0, -1, -1, 0.0, 0.0, 0.0

    Winv = _inv2(W)

    best_x = x0.copy()
    best_obj = 1e300
    best_n = 0
    best_i0 = -1
    best_i1 = -1
    best_l0 = 0.0
    best_l1 = 0.0

    # single actives
    for i in range(m):
        x, lam = _project_one(x0, A[i, 0], A[i, 1], b[i], W, Winv)
        if _is_feasible(x, A, b, tol):
            o = _obj_quadratic(x, x0, W)
            if o < best_obj:
                best_obj = o
                best_x = x
                best_n = 1
                best_i0 = i
                best_i1 = -1
                best_l0 = lam
                best_l1 = 0.0

    # pair actives
    for i in range(m):
        for j in range(i + 1, m):
            Ai = np.empty((2, 2), dtype=np.float64)
            bi = np.empty(2, dtype=np.float64)
            Ai[0, 0] = A[i, 0]
            Ai[0, 1] = A[i, 1]
            Ai[1, 0] = A[j, 0]
            Ai[1, 1] = A[j, 1]
            bi[0] = b[i]
            bi[1] = b[j]

            ok, x, lam = _project_two(x0, Ai, bi, W, Winv)
            if not ok:
                continue
            if _is_feasible(x, A, b, tol):
                o = _obj_quadratic(x, x0, W)
                if o < best_obj:
                    best_obj = o
                    best_x = x
                    best_n = 2
                    best_i0 = i
                    best_i1 = j
                    best_l0 = lam[0]
                    best_l1 = lam[1]

    if best_n > 0:
        return best_x, best_n, best_i0, best_i1, best_l0, best_l1, best_obj

    # fallback sequential projection to find a feasible point
    x = x0.copy()
    for _ in range(25):
        improved = False
        for i in range(m):
            gi = A[i, 0] * x[0] + A[i, 1] * x[1] - b[i]
            if gi > tol:
                x_new, _lam = _project_one(x, A[i, 0], A[i, 1], b[i], W, Winv)
                x = x_new
                improved = True
        if (not improved) and _is_feasible(x, A, b, tol):
            break

    # choose up to 2 "most active" constraints (closest to boundary)
    # by smallest abs(g).
    g0 = 1e300
    g1 = 1e300
    i0 = -1
    i1 = -1
    for i in range(m):
        gi = abs(A[i, 0] * x[0] + A[i, 1] * x[1] - b[i])
        if gi < g0:
            g1 = g0
            i1 = i0
            g0 = gi
            i0 = i
        elif gi < g1:
            g1 = gi
            i1 = i
    # only call as active if close enough
    n_act = 0
    if g0 <= max(tol, 1e-8) and i0 >= 0:
        n_act = 1
    if g1 <= max(tol, 1e-8) and i1 >= 0:
        n_act = 2
    return x, n_act, i0, i1, 0.0, 0.0, _obj_quadratic(x, x0, W)


@dataclass
class ProjectionResult:
    x: np.ndarray
    lam: np.ndarray
    active: np.ndarray
    obj: float


def project_onto_polytope_2d(
    x0: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    W: Optional[np.ndarray] = None,
    tol: float = 1e-10,
) -> ProjectionResult:
    """Closest-point projection onto convex polytope A x <= b in 2D with metric W.

    This is on the hot path for the N–M return mapping. If Numba is available,
    the core search over 1- and 2-active KKT candidates runs JIT-compiled.

    Disable JIT with environment variable:
        DC_USE_NUMBA=0
    """
    x0 = np.asarray(x0, float).reshape(2)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)

    if W is None:
        Wm = np.eye(2, dtype=float)
    else:
        Wm = np.asarray(W, float).reshape(2, 2)

    x, n_act, i0, i1, l0, l1, obj = _project_onto_polytope_2d_kernel(x0, A, b, Wm, float(tol))

    if n_act <= 0:
        active = np.zeros((0,), dtype=int)
        lam = np.zeros((0,), dtype=float)
    elif n_act == 1:
        active = np.array([int(i0)], dtype=int)
        lam = np.array([float(l0)], dtype=float)
    else:
        active = np.array([int(i0), int(i1)], dtype=int)
        lam = np.array([float(l0), float(l1)], dtype=float)

    return ProjectionResult(x=np.asarray(x, float).reshape(2), lam=lam, active=active, obj=float(obj))
