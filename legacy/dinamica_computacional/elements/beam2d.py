from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from dinamica_computacional.core.dof import Node


def rot2d(c: float, s: float) -> np.ndarray:
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    T[:3, :3] = R
    T[3:, 3:] = R
    return T


@dataclass
class Beam2D:
    ni: int
    nj: int
    E: float
    A: float
    I: float
    nodes: List[Node]
    geometry: str = "linear"

    def _geom(self, u: np.ndarray | None = None) -> Tuple[float, float, float]:
        xi, yi = self.nodes[self.ni].x, self.nodes[self.ni].y
        xj, yj = self.nodes[self.nj].x, self.nodes[self.nj].y
        if u is not None:
            ux_i, uy_i = self.nodes[self.ni].dof_u
            ux_j, uy_j = self.nodes[self.nj].dof_u
            xi += float(u[ux_i])
            yi += float(u[uy_i])
            xj += float(u[ux_j])
            yj += float(u[uy_j])
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

    def k_local(self, L: float) -> np.ndarray:
        E, A, I = self.E, self.A, self.I
        k = np.zeros((6, 6))
        k_ax = E * A / L
        k[0, 0] = k_ax; k[0, 3] = -k_ax
        k[3, 0] = -k_ax; k[3, 3] = k_ax

        k11 = 12 * E * I / (L ** 3)
        k12 = 6 * E * I / (L ** 2)
        k22 = 4 * E * I / L
        k22b = 2 * E * I / L

        k[1, 1] = k11;   k[1, 2] = k12;   k[1, 4] = -k11;  k[1, 5] = k12
        k[2, 1] = k12;   k[2, 2] = k22;   k[2, 4] = -k12;  k[2, 5] = k22b
        k[4, 1] = -k11;  k[4, 2] = -k12;  k[4, 4] = k11;   k[4, 5] = -k12
        k[5, 1] = k12;   k[5, 2] = k22b;  k[5, 4] = -k12;  k[5, 5] = k22
        return k

    def k_geo(self, N: float, L: float) -> np.ndarray:
        if abs(N) < 1e-12:
            return np.zeros((6, 6))
        coeff = N / L
        k = np.zeros((6, 6))
        k[1, 1] = 6 / 5
        k[1, 2] = L / 10
        k[1, 4] = -6 / 5
        k[1, 5] = L / 10

        k[2, 1] = L / 10
        k[2, 2] = 2 * L * L / 15
        k[2, 4] = -L / 10
        k[2, 5] = -L * L / 30

        k[4, 1] = -6 / 5
        k[4, 2] = -L / 10
        k[4, 4] = 6 / 5
        k[4, 5] = -L / 10

        k[5, 1] = L / 10
        k[5, 2] = -L * L / 30
        k[5, 4] = -L / 10
        k[5, 5] = 2 * L * L / 15
        return coeff * k

    def stiffness_and_force_global(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        if self.geometry == "corotational":
            L, c, s = self._geom(u)
        else:
            L, c, s = self._geom(None)
        T = rot2d(c, s)
        dofs = self.dofs()
        u_g = u[dofs]
        u_l = T @ u_g
        k_l = self.k_local(L)
        f_l = k_l @ u_l
        k_g = T.T @ k_l @ T

        k_ax = self.E * self.A / L
        du_ax = u_l[3] - u_l[0]
        N_tension = k_ax * du_ax

        if self.geometry == "corotational":
            k_geo = self.k_geo(N_tension, L)
            k_g = T.T @ (k_l + k_geo) @ T

        f_g = T.T @ f_l
        return dofs, k_g, f_g, {"N": float(N_tension)}
