from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .nm_surface import NMSurfacePolygon
from .return_mapping import project_onto_polytope_2d

@dataclass
class PlasticHingeNM:
    """2D plastic hinge in (N,M) with associative flow and return mapping to polygonal N–M surface.

    State variables:
    - q_p: accumulated plastic generalized deformation (2-vector)
    - s: last stress result (N,M)

    K is a 2x2 elastic stiffness in the generalized deformation space:
        s = K (q - q_p)

    For typical usage you can interpret q as:
        q = [epsilon_axial, theta]   (axial strain-like, rotation)
    and K as:
        K = diag(KN, KM)            (axial stiffness, rotational stiffness)

    The constitutive model itself is agnostic to that interpretation, as long as units are consistent.
    """
    surface: NMSurfacePolygon
    K: np.ndarray               # 2x2 elastic stiffness (PD)
    q_p: np.ndarray = None      # 2,
    s: np.ndarray = None        # 2,
    enable_substepping: bool = False
    substep_tol: float = 0.05
    substep_max: int = 12

    def __post_init__(self):
        self.K = np.asarray(self.K, float).reshape(2,2)
        if self.q_p is None:
            self.q_p = np.zeros(2, float)
        else:
            self.q_p = np.asarray(self.q_p, float).reshape(2)
        if self.s is None:
            self.s = np.zeros(2, float)
        else:
            self.s = np.asarray(self.s, float).reshape(2)

    def _update_once_state(
        self,
        s: np.ndarray,
        q_p: np.ndarray,
        dq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Single-step update returning new (s, q_p) and info."""
        dq = np.asarray(dq, float).reshape(2)

        s_trial = s + self.K @ dq

        if self.surface.is_inside(s_trial):
            info = {
                "s_trial": s_trial,
                "s": s_trial.copy(),
                "dq_p_inc": np.zeros(2),
                "q_p": q_p.copy(),
                "active": np.zeros((0,), int),
                "lam": np.zeros((0,)),
                "Kt": self.K.copy(),
            }
            return s_trial, q_p, info

        W = np.linalg.inv(self.K)
        proj = project_onto_polytope_2d(s_trial, self.surface.A, self.surface.b, W=W)
        s_new = proj.x
        dq_p_inc = np.linalg.inv(self.K) @ (s_trial - s_new)

        q_p_new = q_p + dq_p_inc

        if proj.active.size > 0:
            Aact = self.surface.A[proj.active, :]
            denom = Aact @ self.K @ Aact.T
            try:
                denom_inv = np.linalg.inv(denom)
            except np.linalg.LinAlgError:
                denom_inv = np.linalg.pinv(denom)
            Kt = self.K - self.K @ Aact.T @ denom_inv @ Aact @ self.K
            flow_check = Aact.T @ proj.lam
        else:
            Kt = self.K.copy()
            flow_check = np.zeros(2)

        info = {
            "s_trial": s_trial,
            "s": s_new.copy(),
            "dq_p_inc": dq_p_inc,
            "q_p": q_p_new.copy(),
            "active": proj.active.copy(),
            "lam": proj.lam.copy(),
            "flow_check": flow_check,
            "Kt": Kt,
        }
        return s_new, q_p_new, info

    def update(self, dq: np.ndarray, commit: bool = True) -> Dict[str, np.ndarray]:
        """Incremental update with total generalized increment dq (2-vector).

        Returns dict with:
            - s_trial, s, dq_p_inc, q_p
            - active constraints indices, lambdas
        """
        dq = np.asarray(dq, float).reshape(2)

        dq = np.asarray(dq, float).reshape(2)
        s_curr = self.s.copy()
        q_p_curr = self.q_p.copy()

        scale = max(1.0, float(np.max(np.abs(self.surface.b))))
        s_trial = s_curr + self.K @ dq
        f_trial = float(np.max(self.surface.A @ s_trial - self.surface.b))
        nsub = 1
        if self.enable_substepping and f_trial > self.substep_tol * scale:
            nsub = min(self.substep_max, max(2, int(np.ceil(f_trial / (self.substep_tol * scale)))))
        info: Dict[str, np.ndarray] = {}
        for _ in range(nsub):
            s_curr, q_p_curr, info = self._update_once_state(s_curr, q_p_curr, dq / nsub)

        info["nsub"] = np.array([nsub], dtype=int)
        if commit:
            self.s = s_curr
            self.q_p = q_p_curr
            info["s"] = self.s.copy()
            info["q_p"] = self.q_p.copy()
        else:
            info["s"] = s_curr.copy()
            info["q_p"] = q_p_curr.copy()
        return info
