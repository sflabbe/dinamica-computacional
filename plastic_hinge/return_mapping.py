from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

def _solve_kkt_projection(x0: np.ndarray, A: np.ndarray, b: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve projection with equality constraints A x = b (KKT). Returns (x, lambda)."""
    x0 = np.asarray(x0, float).reshape(2)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    W = np.asarray(W, float)

    # KKT:
    # W(x - x0) + A^T λ = 0
    # A x = b
    Winv = np.linalg.inv(W)
    S = A @ Winv @ A.T
    rhs = A @ x0 - b
    lam = np.linalg.solve(S, rhs)
    x = x0 - Winv @ A.T @ lam
    return x, lam

@dataclass
class ProjectionResult:
    x: np.ndarray
    lam: np.ndarray
    active: np.ndarray
    obj: float

def project_onto_polytope_2d(x0: np.ndarray, A: np.ndarray, b: np.ndarray, W: Optional[np.ndarray]=None,
                            tol: float=1e-10) -> ProjectionResult:
    """Closest-point projection onto convex polytope A x <= b in 2D with metric W (PD matrix).

    Minimizes: 0.5 (x-x0)^T W (x-x0)  subject to A x <= b.
    Returns x, multipliers for active constraints, and active indices.

    Notes:
    - For a polyhedral (max) yield function and associative flow, the multipliers define the plastic increment direction.
    - In 2D we can brute-force candidates with 1 or 2 active constraints (active-set / KKT) robustly.
    """
    x0 = np.asarray(x0, float).reshape(2)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)

    if W is None:
        W = np.eye(2)
    else:
        W = np.asarray(W, float).reshape(2,2)

    g = A @ x0 - b
    viol = np.where(g > tol)[0]
    if viol.size == 0:
        return ProjectionResult(x=x0.copy(), lam=np.zeros((0,), float), active=np.zeros((0,), int), obj=0.0)

    # Candidates: project onto each violated constraint (single active), and intersections (two actives)
    candidates: List[ProjectionResult] = []

    # helper objective
    def obj(x):
        d = (x - x0).reshape(2,1)
        return float(0.5 * (d.T @ W @ d))

    # Single active projections
    for i in viol:
        Ai = A[i:i+1,:]
        bi = b[i:i+1]
        x, lam = _solve_kkt_projection(x0, Ai, bi, W)
        if np.all(A @ x - b <= tol + 1e-12) and lam[0] >= -1e-12:
            candidates.append(ProjectionResult(x=x, lam=lam, active=np.array([i], int), obj=obj(x)))

    # Two-active intersections: try combinations among the most violated constraints to limit cost
    # In 2D the true projection will have at most 2 actives.
    # We'll take up to top_k constraints by violation magnitude.
    top_k = min(10, viol.size)
    viol_sorted = viol[np.argsort(g[viol])[::-1]][:top_k]

    for a in range(len(viol_sorted)):
        for bidx in range(a+1, len(viol_sorted)):
            i = int(viol_sorted[a]); j = int(viol_sorted[bidx])
            Aij = A[[i,j], :]
            bij = b[[i,j]]
            # Skip nearly parallel constraints (singular KKT)
            try:
                x, lam = _solve_kkt_projection(x0, Aij, bij, W)
            except np.linalg.LinAlgError:
                continue
            if np.all(A @ x - b <= tol + 1e-12) and np.all(lam >= -1e-12):
                candidates.append(ProjectionResult(x=x, lam=lam, active=np.array([i,j], int), obj=obj(x)))

    if not candidates:
        # Fallback: sequential projection (Dykstra-like one sweep) to get a feasible point
        x = x0.copy()
        for _ in range(25):
            improved = False
            for i in range(A.shape[0]):
                gi = float(A[i] @ x - b[i])
                if gi > tol:
                    # project onto hyperplane Ai x = bi
                    Ai = A[i:i+1,:]; bi = b[i:i+1]
                    x_new, _ = _solve_kkt_projection(x, Ai, bi, W)
                    x = x_new
                    improved = True
            if not improved and np.all(A @ x - b <= tol + 1e-12):
                break
        act = np.where(A @ x - b >= -1e-8)[0]
        lam = np.zeros((act.size,), float)
        return ProjectionResult(x=x, lam=lam, active=act.astype(int), obj=obj(x))

    best = min(candidates, key=lambda r: r.obj)
    return best
