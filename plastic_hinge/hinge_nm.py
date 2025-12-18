from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
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

    def update(self, dq: np.ndarray) -> Dict[str, np.ndarray]:
        """Incremental update with total generalized increment dq (2-vector).

        Returns dict with:
            - s_trial, s, dq_p_inc, q_p
            - active constraints indices, lambdas
        """
        dq = np.asarray(dq, float).reshape(2)

        # elastic predictor
        s_trial = self.s + self.K @ (dq - 0.0)  # s_n + K*dq (plastic part applied in corrector)
        # But we store q_p in stress mapping; use total q via implicit integration:
        # Equivalent: s_trial = K (q_n + dq - q_p_n) = s_n + K*dq
        # Corrector will compute s_{n+1} and update q_p.

        # If inside: accept elastic
        if self.surface.is_inside(s_trial):
            self.s = s_trial
            return {
                "s_trial": s_trial, "s": self.s.copy(),
                "dq_p_inc": np.zeros(2), "q_p": self.q_p.copy(),
                "active": np.zeros((0,), int), "lam": np.zeros((0,))
            }

        # plastic corrector = projection in metric W = K^{-1}
        W = np.linalg.inv(self.K)
        proj = project_onto_polytope_2d(s_trial, self.surface.A, self.surface.b, W=W)
        s_new = proj.x

        # Consistency: s_new = s_trial - K * dq_p_inc  => dq_p_inc = K^{-1}(s_trial - s_new)
        dq_p_inc = np.linalg.inv(self.K) @ (s_trial - s_new)

        # Associative flow for polyhedral surface:
        # dq_p_inc lies in the cone spanned by the active constraint normals.
        # The projection multipliers 'lam' are the KKT multipliers of equality constraints (>=0),
        # and satisfy: W(s_new - s_trial) + A_act^T lam = 0
        # With W=K^{-1}:  (s_trial - s_new) = K A_act^T lam  => dq_p_inc = A_act^T lam
        # We'll report that as a check.
        if proj.active.size > 0:
            Aact = self.surface.A[proj.active, :]
            flow_check = Aact.T @ proj.lam
        else:
            flow_check = np.zeros(2)

        # Update state
        self.q_p = self.q_p + dq_p_inc
        self.s = s_new

        return {
            "s_trial": s_trial, "s": self.s.copy(),
            "dq_p_inc": dq_p_inc, "q_p": self.q_p.copy(),
            "active": proj.active.copy(), "lam": proj.lam.copy(),
            "flow_check": flow_check
        }
