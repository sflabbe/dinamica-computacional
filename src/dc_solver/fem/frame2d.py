"""2D frame elements with optional P-Delta geometric stiffness."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np

from .nodes import Node

# HPC optimization flag: preallocate buffers to reduce allocation pressure
_DC_FAST = os.environ.get("DC_FAST", "0") == "1"


def rot2d(c: float, s: float, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    2D rotation matrix for frame elements (6×6).

    Args:
        c: cos(theta)
        s: sin(theta)
        out: Optional preallocated buffer (6×6) to avoid allocation

    Returns:
        6×6 rotation matrix T
    """
    if out is None:
        T = np.zeros((6, 6))
    else:
        T = out
        T.fill(0.0)

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

    # Beam theory
    # - "euler": Euler–Bernoulli (no shear deformation)
    # - "timoshenko": includes shear deformation via kappa*G*A shear stiffness
    beam_theory: str = "timoshenko"

    # Shear properties for Timoshenko
    nu: float = 0.2          # Poisson's ratio (used to compute G if G is None)
    kappa: float = 5.0 / 6.0 # shear correction factor
    G: float | None = None   # shear modulus; if None, computed from E and nu

    def __post_init__(self):
        """Initialize preallocated buffers for fast mode (DC_FAST=1)."""
        if _DC_FAST:
            # Preallocate reusable buffers to reduce allocation pressure
            self._buf_T = np.zeros((6, 6), dtype=float)
            self._buf_k_local = np.zeros((6, 6), dtype=float)
            self._buf_k_global = np.zeros((6, 6), dtype=float)
            self._buf_u_local = np.zeros(6, dtype=float)
            self._buf_f_local = np.zeros(6, dtype=float)
            self._buf_f_global = np.zeros(6, dtype=float)
        else:
            self._buf_T = None
            self._buf_k_local = None
            self._buf_k_global = None
            self._buf_u_local = None
            self._buf_f_local = None
            self._buf_f_global = None

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

    def k_local(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Local element stiffness matrix (6×6).

        Args:
            out: Optional preallocated buffer (6×6) to avoid allocation

        Returns:
            6×6 local stiffness matrix
        """
        L, _, _ = self._geom()
        E, A, I = self.E, self.A, self.I

        if out is None:
            k = np.zeros((6, 6))
        else:
            k = out
            k.fill(0.0)

        # axial (same for Euler/Timoshenko)
        k_ax = E * A / L
        k[0, 0] = k_ax
        k[0, 3] = -k_ax
        k[3, 0] = -k_ax
        k[3, 3] = k_ax

        theory = (self.beam_theory or "euler").lower().strip()
        if theory in ("timo", "timoshenko", "shear"):
            # Timoshenko bending-shear stiffness
            # phi = 12EI / (kappa*G*A*L^2)
            G = self.G if self.G is not None else (E / (2.0 * (1.0 + float(self.nu))))
            As = float(self.kappa) * A
            if As <= 0.0 or G <= 0.0:
                raise ValueError("Timoshenko requires positive shear stiffness (kappa*A and G).")

            phi = 12.0 * E * I / (G * As * (L ** 2))

            k11 = 12.0 * E * I / ((L ** 3) * (1.0 + phi))
            k12 = 6.0 * E * I / ((L ** 2) * (1.0 + phi))
            k22 = (4.0 + phi) * E * I / (L * (1.0 + phi))
            k22b = (2.0 - phi) * E * I / (L * (1.0 + phi))
        else:
            # Euler–Bernoulli bending stiffness
            k11 = 12.0 * E * I / (L ** 3)
            k12 = 6.0 * E * I / (L ** 2)
            k22 = 4.0 * E * I / L
            k22b = 2.0 * E * I / L

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
        """
        Global element stiffness matrix and force vector.

        Uses preallocated buffers when DC_FAST=1 to reduce allocation pressure.

        Args:
            u: Global displacement vector
            include_geo: Include geometric stiffness (P-Delta) if True

        Returns:
            (dofs, k_global, f_global, metadata)
        """
        L, c, s = self._geom()

        # Use preallocated buffers if available (DC_FAST=1)
        if _DC_FAST and self._buf_T is not None:
            T = rot2d(c, s, out=self._buf_T)
            k_l = self.k_local(out=self._buf_k_local)

            dofs = self.dofs()
            u_g = u[dofs]

            # Reuse buffers for u_local and f_local
            np.dot(T, u_g, out=self._buf_u_local)
            u_l = self._buf_u_local
            np.dot(k_l, u_l, out=self._buf_f_local)
            f_l = self._buf_f_local

            # Material (Timoshenko) contribution
            # k_g = T.T @ k_l @ T
            np.dot(k_l, T, out=self._buf_k_global)  # temp = k_l @ T
            np.dot(T.T, self._buf_k_global, out=self._buf_k_global)  # k_g = T.T @ temp
            k_g = self._buf_k_global

            # f_g = T.T @ f_l
            np.dot(T.T, f_l, out=self._buf_f_global)
            f_g = self._buf_f_global
        else:
            # Standard path (no buffer optimization)
            T = rot2d(c, s)
            dofs = self.dofs()
            u_g = u[dofs]
            u_l = T @ u_g
            k_l = self.k_local()
            f_l = k_l @ u_l

            # Material (Timoshenko) contribution
            k_g = T.T @ k_l @ T
            f_g = T.T @ f_l

        # axial force in tension-positive convention
        k_ax = self.E * self.A / L
        du_ax = u_l[3] - u_l[0]
        N_tension = k_ax * du_ax

        if include_geo:
            k_geo_l = self.k_geo_local(N_tension)
            if _DC_FAST and self._buf_T is not None:
                # k_g += T.T @ k_geo_l @ T
                temp = np.dot(k_geo_l, T)
                k_g += np.dot(T.T, temp)
                # f_g += T.T @ (k_geo_l @ u_l)
                f_g += np.dot(T.T, np.dot(k_geo_l, u_l))
            else:
                k_g += T.T @ k_geo_l @ T
                f_g += T.T @ (k_geo_l @ u_l)

        # Return copies to ensure caller can mutate without affecting buffers
        if _DC_FAST and self._buf_T is not None:
            return dofs, k_g.copy(), f_g.copy(), {"N": float(N_tension)}
        else:
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

    def end_forces_local(self, u: np.ndarray) -> Dict[str, float]:
        """Return element end forces in local coordinates."""
        L, c, s = self._geom()
        _ = L
        T = rot2d(c, s)
        dofs = self.dofs()
        u_g = u[dofs]
        u_l = T @ u_g
        f_l = self.k_local() @ u_l
        return {
            "N": float(f_l[0]),
            "Vi": float(f_l[1]),
            "Mi": float(f_l[2]),
            "Vj": float(f_l[4]),
            "Mj": float(f_l[5]),
        }
