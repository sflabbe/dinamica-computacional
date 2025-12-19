"""Hinge constitutive models and rotational spring element."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from plastic_hinge import NMSurfacePolygon
from dc_solver.fem.nodes import Node


def moment_capacity_from_polygon(surface: NMSurfacePolygon, N: float) -> float:
    """Return max |M| for a given axial N by intersecting the convex polygon with N=const."""
    V = np.asarray(surface.vertices, float)
    Nmin, Nmax = float(np.min(V[:, 0])), float(np.max(V[:, 0]))
    Nc = float(np.clip(N, Nmin, Nmax))

    Ms = []
    for i in range(V.shape[0]):
        a = V[i]
        b = V[(i + 1) % V.shape[0]]
        Na, Ma = float(a[0]), float(a[1])
        Nb, Mb = float(b[0]), float(b[1])
        if (Na - Nc) == 0.0 and (Nb - Nc) == 0.0:
            Ms.extend([Ma, Mb])
            continue
        if (Na - Nc) * (Nb - Nc) > 0:
            continue
        if abs(Nb - Na) < 1e-18:
            continue
        t = (Nc - Na) / (Nb - Na)
        if -1e-12 <= t <= 1.0 + 1e-12:
            Mi = Ma + t * (Mb - Ma)
            Ms.append(float(Mi))

    if len(Ms) == 0:
        j = int(np.argmin(np.abs(V[:, 0] - Nc)))
        return float(abs(V[j, 1]))
    return float(max(abs(min(Ms)), abs(max(Ms))))


@dataclass
class ColumnHingeNMRot:
    """Moment-rotation hinge with My dependent on an axial reference N_ref."""

    surface: NMSurfacePolygon
    k0: float
    alpha_post: float = 1e-4

    th_p_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0
    My_comm: float = 1.0

    def set_yield_from_N(self, N_ref: float) -> None:
        self.My_comm = max(1e-6, moment_capacity_from_polygon(self.surface, N_ref))

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float]:
        K0 = float(self.k0)
        k_reg = max(float(self.alpha_post) * K0, 1e-12)

        M_trial = float(self.M_comm) + K0 * float(dth)
        f = abs(M_trial) - float(self.My_comm)

        if f <= 0.0:
            return M_trial, K0, self.th_p_comm, self.a_comm, M_trial

        dg = f / max(K0, 1e-18)
        sgn = 1.0 if M_trial >= 0.0 else -1.0
        th_p_new = float(self.th_p_comm) + dg * sgn
        a_new = float(self.a_comm) + dg
        M_new = sgn * float(self.My_comm)

        k_tan = k_reg
        return M_new, k_tan, th_p_new, a_new, M_new


@dataclass
class SHMBeamHinge1D:
    K0_0: float
    My_0: float
    alpha_post: float = 0.02
    cK: float = 2.0
    cMy: float = 1.0

    th_p_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float]:
        K0 = self.K0_0 * math.exp(-self.cK * self.a_comm)
        My = self.My_0 * math.exp(-self.cMy * self.a_comm)
        Kp = self.alpha_post * K0
        H = (K0 * Kp) / max(K0 - Kp, 1e-18)

        M_trial = self.M_comm + K0 * dth
        f = abs(M_trial) - (My + H * self.a_comm)
        if f <= 0.0:
            return M_trial, K0, self.th_p_comm, self.a_comm, M_trial

        dg = f / (K0 + H)
        sgn = 1.0 if M_trial >= 0 else -1.0
        th_p_new = self.th_p_comm + dg * sgn
        a_new = self.a_comm + dg
        M_new = M_trial - K0 * dg * sgn
        k_tan = (K0 * H) / (K0 + H)
        return M_new, k_tan, th_p_new, a_new, M_new


@dataclass
class RotSpringElement:
    """Zero-length rotational spring between node i and j (only θ DOFs)."""

    ni: int
    nj: int
    kind: str  # "col_nm" or "beam_shm"
    col_hinge: Optional[ColumnHingeNMRot]
    beam_hinge: Optional[SHMBeamHinge1D]
    nodes: List[Node]

    _trial: Dict | None = None

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        dofs = self.dofs()
        th_i = float(u_trial[dofs[2]])
        th_j = float(u_trial[dofs[5]])
        th_i_c = float(u_comm[dofs[2]])
        th_j_c = float(u_comm[dofs[5]])

        dth_inc = (th_j - th_i) - (th_j_c - th_i_c)

        if self.kind == "col_nm":
            assert self.col_hinge is not None
            a_old = float(self.col_hinge.a_comm)
            th_p_old = float(self.col_hinge.th_p_comm)
            M_old = float(self.col_hinge.M_comm)
            My_old = float(self.col_hinge.My_comm)

            M, kM, th_p_new, a_new, M_new = self.col_hinge.eval_increment(dth_inc)
            dW_pl = max(0.0, 0.5 * (M_old + M) * (th_p_new - th_p_old))
            info_extra = {"My": float(My_old), "a": float(a_old), "dW_pl": float(dW_pl)}
        elif self.kind == "beam_shm":
            assert self.beam_hinge is not None
            a_old = float(self.beam_hinge.a_comm)
            th_p_old = float(self.beam_hinge.th_p_comm)
            M_old = float(self.beam_hinge.M_comm)

            M, kM, th_p_new, a_new, M_new = self.beam_hinge.eval_increment(dth_inc)
            dW_pl = max(0.0, 0.5 * (M_old + M) * (th_p_new - th_p_old))
            info_extra = {"a": float(a_old), "dW_pl": float(dW_pl)}
        else:
            raise ValueError("Unknown hinge kind")

        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = kM * (Bm.T @ Bm)
        f_l = (Bm.T * M).reshape(6)

        self._trial = {
            "th_p_new": th_p_new,
            "a_new": a_new,
            "M_new": M_new,
        }
        info = {"dtheta": float(dth_inc), "M": float(M)}
        info.update(info_extra)
        return k_l, f_l, info

    def commit(self) -> None:
        if self._trial is None:
            return
        if self.kind == "col_nm":
            assert self.col_hinge is not None
            self.col_hinge.th_p_comm = self._trial["th_p_new"]
            self.col_hinge.a_comm = self._trial["a_new"]
            self.col_hinge.M_comm = self._trial["M_new"]
        elif self.kind == "beam_shm":
            assert self.beam_hinge is not None
            self.beam_hinge.th_p_comm = self._trial["th_p_new"]
            self.beam_hinge.a_comm = self._trial["a_new"]
            self.beam_hinge.M_comm = self._trial["M_new"]
