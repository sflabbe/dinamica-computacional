"""2D frame elements with optional P-Delta geometric stiffness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from .nodes import Node


def rot2d(c: float, s: float) -> np.ndarray:
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


@dataclass
class FrameElementLinear2D:
    ni: int
    nj: int
    E: float
    A: float
    I: float
    nodes: List[Node]

    def _geom(self) -> Tuple[float, float, float]:
        xi, yi = self.nodes[self.ni].x, self.nodes[self.ni].y
        xj, yj = self.nodes[self.nj].x, self.nodes[self.nj].y
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        c, s = dx / L, dy / L
        return L, c, s

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def k_local(self) -> np.ndarray:
        L, _, _ = self._geom()
        E, A, I = self.E, self.A, self.I
        k = np.zeros((6, 6))
        k_ax = E * A / L
        k[0, 0] = k_ax
        k[0, 3] = -k_ax
        k[3, 0] = -k_ax
        k[3, 3] = k_ax

        k11 = 12 * E * I / (L ** 3)
        k12 = 6 * E * I / (L ** 2)
        k22 = 4 * E * I / L
        k22b = 2 * E * I / L

        k[1, 1] = k11
        k[1, 2] = k12
        k[1, 4] = -k11
        k[1, 5] = k12
        k[2, 1] = k12
        k[2, 2] = k22
        k[2, 4] = -k12
        k[2, 5] = k22b
        k[4, 1] = -k11
        k[4, 2] = -k12
        k[4, 4] = k11
        k[4, 5] = -k12
        k[5, 1] = k12
        k[5, 2] = k22b
        k[5, 4] = -k12
        k[5, 5] = k22
        return k

    def k_geo_local(self, N: float) -> np.ndarray:
        """Small-rotation geometric stiffness (P-Delta) in local coordinates."""
        L, _, _ = self._geom()
        k = np.zeros((6, 6))
        if abs(L) < 1e-12:
            return k
        coeff = N / (30.0 * L)
        k_geo = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 36.0, 3.0 * L, 0.0, -36.0, 3.0 * L],
            [0.0, 3.0 * L, 4.0 * L * L, 0.0, -3.0 * L, -1.0 * L * L],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -36.0, -3.0 * L, 0.0, 36.0, -3.0 * L],
            [0.0, 3.0 * L, -1.0 * L * L, 0.0, -3.0 * L, 4.0 * L * L],
        ])
        k = coeff * k_geo
        return k

    def stiffness_and_force_global(
        self,
        u: np.ndarray,
        include_geo: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        L, c, s = self._geom()
        T = rot2d(c, s)
        dofs = self.dofs()
        u_g = u[dofs]
        u_l = T @ u_g
        k_l = self.k_local()
        f_l = k_l @ u_l

        k_g = T.T @ k_l @ T
        f_g = T.T @ f_l

        # axial force in tension-positive convention
        k_ax = self.E * self.A / L
        du_ax = u_l[3] - u_l[0]
        N_tension = k_ax * du_ax

        if include_geo:
            k_geo_l = self.k_geo_local(N_tension)
            k_g += T.T @ k_geo_l @ T

        return dofs, k_g, f_g, {"N": float(N_tension)}

    def equiv_nodal_load_global(self, w_global: Tuple[float, float]) -> np.ndarray:
        """Consistent nodal load for uniform distributed load in global axes."""
        L, c, s = self._geom()
        gx, gy = w_global
        wx = c * gx + s * gy
        wy = -s * gx + c * gy

        f_l = np.zeros(6, dtype=float)
        # axial (local x)
        f_l[0] += wx * L / 2.0
        f_l[3] += wx * L / 2.0
        # transverse (local y)
        f_l[1] += wy * L / 2.0
        f_l[2] += wy * L * L / 12.0
        f_l[4] += wy * L / 2.0
        f_l[5] += -wy * L * L / 12.0

        T = rot2d(c, s)
        return T.T @ f_l
