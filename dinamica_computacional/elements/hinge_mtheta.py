from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from dinamica_computacional.core.dof import Node


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
class RotSpringElementMTheta:
    ni: int
    nj: int
    hinge: SHMBeamHinge1D
    nodes: list[Node]
    _trial: Optional[Dict] = None

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array([
            ni.dof_u[0], ni.dof_u[1], ni.dof_th,
            nj.dof_u[0], nj.dof_u[1], nj.dof_th,
        ], dtype=int)

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray):
        dofs = self.dofs()
        th_i = float(u_trial[dofs[2]])
        th_j = float(u_trial[dofs[5]])
        th_i_c = float(u_comm[dofs[2]])
        th_j_c = float(u_comm[dofs[5]])

        dth_inc = (th_j - th_i) - (th_j_c - th_i_c)

        a_old = float(self.hinge.a_comm)
        th_p_old = float(self.hinge.th_p_comm)
        M_old = float(self.hinge.M_comm)

        K0 = float(self.hinge.K0_0 * math.exp(-self.hinge.cK * a_old))
        My = float(self.hinge.My_0 * math.exp(-self.hinge.cMy * a_old))

        M, kM, th_p_new, a_new, M_new = self.hinge.eval_increment(dth_inc)
        trial_state = {"th_p_new": th_p_new, "a_new": a_new, "M_new": M_new, "M": float(M)}

        dg = float(a_new) - a_old
        dth_p = float(th_p_new) - th_p_old
        dW_pl = abs(float(M_new)) * dg

        info_extra = {
            "My": My,
            "K0": K0,
            "dg": dg,
            "dtheta_p": dth_p,
            "M_old": M_old,
            "M_new": float(M_new),
            "a": float(a_new),
            "th_p": float(th_p_new),
            "dW_pl": float(dW_pl),
        }

        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = kM * (Bm.T @ Bm)
        f_l = (Bm.T * M).reshape(6)

        self._trial = trial_state
        info = {"dtheta": float(dth_inc), "M": float(M)}
        info.update(info_extra)
        return k_l, f_l, info

    def commit(self) -> None:
        if self._trial is None:
            return
        self.hinge.th_p_comm = self._trial["th_p_new"]
        self.hinge.a_comm = self._trial["a_new"]
        self.hinge.M_comm = self._trial["M_new"]

    def reset(self) -> None:
        self._trial = None
        self.hinge.th_p_comm = 0.0
        self.hinge.a_comm = 0.0
        self.hinge.M_comm = 0.0
