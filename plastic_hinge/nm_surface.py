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

    def simplify(
        self,
        *,
        angle_tol_rad: float = 0.01,
        max_vertices: int | None = 80,
    ) -> "NMSurfacePolygon":
        """Return a simplified polygon by removing nearly-collinear vertices.

        Motivation
        ----------
        A convex hull built from many sampled points can contain a very large
        number of short facets. In return-mapping with an active-set on polygon
        facets this may cause "zig-zag" (serrucho) responses when the active
        edge switches frequently between adjacent facets.

        This routine removes vertices with a very small turning angle (nearly
        collinear), and optionally enforces a maximum vertex count by dropping
        the smallest turning angles.

        Parameters
        ----------
        angle_tol_rad : float
            Threshold on sin(turning angle). Smaller => keeps more points.
        max_vertices : int | None
            If not None, enforce an upper bound on number of vertices.
        """

        verts = np.asarray(self.vertices, float)
        if verts.shape[0] <= 3:
            return self

        def _sin_turning(i: int, vv: np.ndarray) -> float:
            k = vv.shape[0]
            p = vv[(i - 1) % k]
            c = vv[i % k]
            n = vv[(i + 1) % k]
            e1 = c - p
            e2 = n - c
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 <= 0.0 or n2 <= 0.0:
                return 0.0
            cross = abs(e1[0] * e2[1] - e1[1] * e2[0])
            return float(cross / (n1 * n2))

        # 1) Drop nearly-collinear vertices.
        keep_mask = np.ones(verts.shape[0], dtype=bool)
        sin_thr = float(np.sin(angle_tol_rad))
        for i in range(verts.shape[0]):
            if _sin_turning(i, verts) < sin_thr:
                keep_mask[i] = False
        verts2 = verts[keep_mask]
        if verts2.shape[0] < 3:
            verts2 = verts

        # 2) Enforce max_vertices by removing the smallest turning angles.
        if max_vertices is not None and verts2.shape[0] > int(max_vertices):
            vv = verts2
            while vv.shape[0] > int(max_vertices) and vv.shape[0] > 3:
                sins = np.array([_sin_turning(i, vv) for i in range(vv.shape[0])], dtype=float)
                idx = int(np.argmin(sins))
                vv = np.delete(vv, idx, axis=0)
            verts2 = vv

        A, b = polygon_halfspaces_ccw(verts2)
        return NMSurfacePolygon(vertices=verts2, A=A, b=b)
