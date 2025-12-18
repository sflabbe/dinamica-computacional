from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np

def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[1] - a[1]*b[0])

def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain convex hull. Returns points in CCW order (no duplicate last point)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 1:
        return pts.copy()

    # sort by x then y
    idx = np.lexsort((pts[:,1], pts[:,0]))
    pts = pts[idx]

    def build_half(pts_half):
        hull = []
        for p in pts_half:
            p = np.asarray(p, float)
            while len(hull) >= 2:
                a = hull[-2]; b = hull[-1]
                if cross2d(b-a, p-b) <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)
        return hull

    lower = build_half(pts)
    upper = build_half(pts[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) == 0:
        return np.zeros((0,2), float)
    return np.vstack(hull)

def polygon_halfspaces_ccw(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given CCW polygon vertices, returns (A,b) such that inside satisfies A x <= b."""
    p = np.asarray(poly, float)
    n = p.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for a polygon.")
    A = np.zeros((n,2), float)
    b = np.zeros((n,), float)
    for i in range(n):
        pi = p[i]
        pj = p[(i+1) % n]
        e = pj - pi
        # For CCW polygon: inside is LEFT of each directed edge.
        # Condition: dot(n_left, x - pi) >= 0, with n_left = [-e_y, e_x]
        # Convert to <= form by multiplying by -1:
        a = np.array([e[1], -e[0]], float)  # -n_left
        A[i,:] = a
        b[i] = float(a @ pi)
    return A, b

def point_in_halfspaces(x: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float=1e-12) -> bool:
    x = np.asarray(x, float).reshape(2)
    return bool(np.all(A @ x - b <= tol))
