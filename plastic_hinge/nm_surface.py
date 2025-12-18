from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .geometry import convex_hull, polygon_halfspaces_ccw, point_in_halfspaces

@dataclass
class NMSurfacePolygon:
    """Convex polygonal yield surface in (N,M) with half-space form A s <= b."""
    vertices: np.ndarray  # (k,2) CCW
    A: np.ndarray         # (k,2)
    b: np.ndarray         # (k,)

    @classmethod
    def from_points(cls, points_nm: np.ndarray) -> "NMSurfacePolygon":
        pts = np.asarray(points_nm, float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points_nm must be (n,2) array of [N, M].")
        hull = convex_hull(pts)
        if hull.shape[0] < 3:
            raise ValueError("Not enough non-collinear points to build a polygon.")
        A, b = polygon_halfspaces_ccw(hull)
        return cls(vertices=hull, A=A, b=b)

    def is_inside(self, s_nm: np.ndarray, tol: float=1e-10) -> bool:
        return point_in_halfspaces(s_nm, self.A, self.b, tol=tol)

    def yield_value(self, s_nm: np.ndarray) -> float:
        s = np.asarray(s_nm, float).reshape(2)
        return float(np.max(self.A @ s - self.b))

    def active_set(self, s_nm: np.ndarray, tol: float=1e-10) -> np.ndarray:
        s = np.asarray(s_nm, float).reshape(2)
        g = self.A @ s - self.b
        return np.where(g >= -tol)[0]
