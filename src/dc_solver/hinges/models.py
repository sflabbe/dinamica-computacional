"""Hinge constitutive models and rotational spring element."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from plastic_hinge import NMSurfacePolygon, PlasticHingeNM
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
    """Simplified Smooth Hysteretic Model (SHM) hinge with degradation and pinching."""

    K0_0: float
    My_0: float
    alpha_post: float = 0.02
    cK: float = 2.0
    cMy: float = 1.0
    bw_A: float = 1.0
    bw_beta: float = 0.6
    bw_gamma: float = 0.4
    bw_n: float = 2.0
    pinch: float = 0.3
    theta_pinch: float = 0.002

    th_comm: float = 0.0
    z_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0

    def _bw_rhs(self, z: float, sign_dth: float) -> float:
        return self.bw_A - self.bw_beta * sign_dth * abs(z) ** (self.bw_n - 1.0) * z - self.bw_gamma * abs(z) ** self.bw_n

    def eval_increment(self, dth: float, nsub: int | None = None) -> Tuple[float, float, float, float, float]:
        if nsub is None:
            nsub = max(1, int(np.ceil(abs(dth) / 2e-4)))
        dth_sub = dth / nsub
        th = self.th_comm
        z = self.z_comm
        a = self.a_comm
        M = self.M_comm

        for _ in range(nsub):
            th_new = th + dth_sub
            sign_dth = 1.0 if dth_sub >= 0.0 else -1.0

            def rhs(z_local: float) -> float:
                return self._bw_rhs(z_local, sign_dth)

            k1 = rhs(z)
            k2 = rhs(z + 0.5 * dth_sub * k1)
            k3 = rhs(z + 0.5 * dth_sub * k2)
            k4 = rhs(z + dth_sub * k3)
            z_new = z + (dth_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            K0 = self.K0_0 * math.exp(-self.cK * a)
            My = self.My_0 * math.exp(-self.cMy * a)
            pinch_factor = 1.0 - self.pinch * math.exp(-abs(th_new) / max(self.theta_pinch, 1e-12))
            M_new = self.alpha_post * K0 * th_new + (1.0 - self.alpha_post) * My * z_new * pinch_factor

            a += max(0.0, 0.5 * (M + M_new) * dth_sub)
            th = th_new
            z = max(-1.2, min(1.2, z_new))
            M = M_new

        K0 = self.K0_0 * math.exp(-self.cK * a)
        My = self.My_0 * math.exp(-self.cMy * a)
        k_tan = self.alpha_post * K0 + (1.0 - self.alpha_post) * My * (1.0 - self.pinch)
        return M, k_tan, th, a, M



@dataclass
class SHMSivaselvanReinhorn1D:
    """
    Smooth Hysteretic Model (SHM) 1D hinge with stiffness/strength deterioration
    following Sivaselvan & Reinhorn (2000), Eqs. (6), (8) and energy update (9).

    Notes
    -----
    - Uses a simplified explicit integration of Eq. (5): M_{n+1} = M_n + K_t(M_n, th_n, dth) * dth.
    - Greek alpha (``alpha_pivot``) controls stiffness deterioration (pivot rule).
    - beta1 and beta2 control strength deterioration per Eq. (8).
    - mu_ult defines ultimate ductility: th_u = mu_ult * th_y0.
    - The remaining SHM parameters (N, eta1, eta2, a_post) are kept with reasonable defaults;
      tune them if you need a closer match to the published curves.
    """

    # Backbone / initial properties
    K0: float
    My0: float

    # SHM "post-yield spring" ratio (latin a in the paper)
    a_post: float = 0.02

    # Smooth transition / unloading-shape parameters (Eq. (3))
    N_smooth: float = 10.0
    eta1: float = 0.5
    eta2: float = 0.5

    # Deterioration parameters shown in Fig. 4 (SAC 1996)
    alpha_pivot: float = 10.0   # Greek alpha in Eq. (6)
    beta1: float = 0.60         # ductility-based strength deterioration (Eq. 8)
    beta2: float = 0.30         # energy-based strength deterioration (Eq. 8)
    mu_ult: float = 8.0         # ultimate ductility = th_u / th_y0

    # Optional minimal pinching (not part of Fig. 4 parameters)
    pinch: float = 0.0
    theta_pinch: float = 0.002

    # Optional: override H_ult (if None, a heuristic value is computed)
    Hult: float | None = None

    # Committed state
    th_comm: float = 0.0
    M_comm: float = 0.0
    H_comm: float = 0.0
    th_max_pos_comm: float = 0.0
    th_max_neg_comm: float = 0.0

    # Derived committed yields (magnitudes)
    My_pos_comm: float | None = None
    My_neg_comm: float | None = None

    def _th_y0(self) -> float:
        return float(abs(self.My0) / max(abs(self.K0), 1e-18))

    def _th_u(self) -> float:
        return float(self.mu_ult * self._th_y0())

    def _H_ult(self) -> float:
        # Heuristic: characteristic energy scale at ultimate ductility.
        # Users may override via Hult=...
        if self.Hult is not None:
            return float(max(self.Hult, 1e-18))
        return float(max(abs(self.My0) * self._th_u(), 1e-18))

    def _strength_factor(self, th_max_mag: float, H: float) -> float:
        th_u = self._th_u()
        # Envelope (ductility-based) term: [1 - (th_max/th_u)^(1/beta1)]
        if th_u <= 0.0:
            env = 1.0
        else:
            ratio = min(max(th_max_mag / th_u, 0.0), 1.0)
            b1 = max(float(self.beta1), 1e-12)
            env = 1.0 - (ratio ** (1.0 / b1))
            env = max(0.0, env)

        # Energy-based term: [1 - beta2/(1-beta2) * H/Hult]
        b2 = float(self.beta2)
        if b2 <= 0.0:
            en = 1.0
        elif b2 >= 0.999:
            en = 0.0
        else:
            en = 1.0 - (b2 / (1.0 - b2)) * (float(H) / self._H_ult())
            en = max(0.0, en)

        return env * en

    def _update_yields(self, th_max_pos: float, th_max_neg: float, H: float) -> tuple[float, float]:
        # Eq. (8) applied separately for positive and negative sides (symmetric magnitudes).
        f_pos = self._strength_factor(abs(th_max_pos), H)
        f_neg = self._strength_factor(abs(th_max_neg), H)
        return abs(self.My0) * f_pos, abs(self.My0) * f_neg

    def _Rk(self, M_mag: float, th_mag: float, My_mag: float) -> float:
        # Eq. (6): K_cur = Rk*K0 = (M_cur + α My)/(K0 th_cur + α My) * K0
        # -> Rk = (M + α My) / (K0 th + α My)
        num = float(M_mag + self.alpha_pivot * My_mag)
        den = float(abs(self.K0) * th_mag + self.alpha_pivot * My_mag)
        if den <= 1e-18:
            return 1.0
        rk = num / den
        return max(1e-6, rk)

    def eval_increment(self, dth: float) -> tuple[float, float, float, float, float]:
        """
        Advance hinge state by increment dth.

        Returns
        -------
        M_new : float
        k_tan : float
        th_new : float
        H_new : float
        M_comm_out : float
        """
        dth = float(dth)
        th_new = float(self.th_comm + dth)

        # Update maxima (used by strength degradation)
        th_max_pos = max(float(self.th_max_pos_comm), th_new)
        th_max_neg = min(float(self.th_max_neg_comm), th_new)

        # Update degraded yields using committed energy (explicit)
        My_pos, My_neg = self._update_yields(th_max_pos, th_max_neg, float(self.H_comm))
        self.My_pos_comm = My_pos
        self.My_neg_comm = My_neg

        # Select active yield magnitude by deformation side (Eq. 4 uses sgn(th))
        My_act = My_pos if th_new >= 0.0 else My_neg
        My_star = max((1.0 - float(self.a_post)) * float(My_act), 1e-18)

        # Stiffness degradation factor Rk (Eq. 6), evaluated at committed state
        rk = self._Rk(abs(self.M_comm), abs(self.th_comm), float(My_act))
        K_cur = rk * float(abs(self.K0))

        # Smooth hysteretic tangent contribution (Eq. 3 / 7)
        th_c = float(self.th_comm)
        M_c = float(self.M_comm)

        # Portion carried by post-yield spring
        M_post_c = float(self.a_post) * float(self.K0) * th_c
        M_star_c = M_c - M_post_c

        # Loading/unloading indicator uses sgn(M* dth)
        sgn = 1.0 if (M_star_c * dth) >= 0.0 else -1.0
        shape = float(self.eta1) * sgn + float(self.eta2)

        ratio = abs(M_star_c) / My_star
        ratio = min(max(ratio, 0.0), 2.0)  # mild cap to avoid overflow
        Np = max(float(self.N_smooth), 1.0)
        smooth_term = 1.0 - (ratio ** Np) * shape
        # Avoid negative stiffness
        smooth_term = max(0.0, smooth_term)

        # Eq. (7): K_hysteretic = (Rk - a_post) K0 * smooth_term
        K_hyst = (rk - float(self.a_post)) * float(self.K0) * smooth_term

        # Optional minimal pinching (simple multiplier near origin)
        if self.pinch > 0.0:
            pinch_factor = 1.0 - float(self.pinch) * math.exp(-abs(th_new) / max(float(self.theta_pinch), 1e-12))
        else:
            pinch_factor = 1.0

        # Total tangent stiffness (approx)
        k_tan = float(self.a_post) * float(self.K0) + K_hyst
        k_tan_eff = k_tan * pinch_factor

        # Explicit update of moment
        M_new = M_c + k_tan_eff * dth

        # Energy update (Eq. 9) — accumulate dissipated hysteretic energy
        dM = M_new - M_c
        avgM = 0.5 * (M_c + M_new)
        # ΔH = [ (M + (M+ΔM))/2 ] (Δth - ΔM/(Rk K0))  (use K_cur = Rk*K0)
        dH = avgM * (dth - (dM / max(K_cur, 1e-18)))
        H_new = float(self.H_comm + abs(dH))

        # Return + also the "committed" M (consistent with existing interface)
        return M_new, k_tan_eff, th_new, H_new, M_new


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
            th_next = th_p_new
        elif self.kind == "beam_shm":
            assert self.beam_hinge is not None
            a_old = float(self.beam_hinge.a_comm)
            th_old = float(self.beam_hinge.th_comm)
            M_old = float(self.beam_hinge.M_comm)

            M, kM, th_new, a_new, M_new = self.beam_hinge.eval_increment(dth_inc)
            dW_pl = max(0.0, 0.5 * (M_old + M) * (th_new - th_old))
            info_extra = {"a": float(a_old), "dW_pl": float(dW_pl)}
            th_next = th_new
        else:
            raise ValueError("Unknown hinge kind")

        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = kM * (Bm.T @ Bm)
        f_l = (Bm.T * M).reshape(6)

        self._trial = {
            "th_p_new": th_next,
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
            self.beam_hinge.th_comm = self._trial["th_p_new"]
            self.beam_hinge.a_comm = self._trial["a_new"]
            self.beam_hinge.M_comm = self._trial["M_new"]

    def reset_state(self) -> None:
        self._trial = None
        if self.kind == "col_nm":
            assert self.col_hinge is not None
            self.col_hinge.th_p_comm = 0.0
            self.col_hinge.a_comm = 0.0
            self.col_hinge.M_comm = 0.0
        elif self.kind == "beam_shm":
            assert self.beam_hinge is not None
            self.beam_hinge.th_comm = 0.0
            self.beam_hinge.a_comm = 0.0
            self.beam_hinge.M_comm = 0.0


@dataclass
class ColumnHingeNM2D:
    """Wrapper for N-M hinge with associative flow and consistent tangent."""

    hinge: PlasticHingeNM


@dataclass
class HingeNM2DElement:
    """Zero-length N-M hinge between node i and j (axial + rotation)."""

    ni: int
    nj: int
    hinge: ColumnHingeNM2D
    nodes: List[Node]

    _trial: Dict | None = None

    def dofs(self) -> np.ndarray:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        return np.array(
            [
                ni.dof_u[0],
                ni.dof_u[1],
                ni.dof_th,
                nj.dof_u[0],
                nj.dof_u[1],
                nj.dof_th,
            ],
            dtype=int,
        )

    def _axis(self) -> Tuple[float, float]:
        ni = self.nodes[self.ni]
        nj = self.nodes[self.nj]
        dx = float(nj.x - ni.x)
        dy = float(nj.y - ni.y)
        L = math.hypot(dx, dy)
        if L <= 1e-12:
            return 1.0, 0.0
        return dx / L, dy / L

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        dofs = self.dofs()
        cx, cy = self._axis()

        u_i = u_trial[dofs[0]]
        v_i = u_trial[dofs[1]]
        th_i = u_trial[dofs[2]]
        u_j = u_trial[dofs[3]]
        v_j = u_trial[dofs[4]]
        th_j = u_trial[dofs[5]]

        u_i_c = u_comm[dofs[0]]
        v_i_c = u_comm[dofs[1]]
        th_i_c = u_comm[dofs[2]]
        u_j_c = u_comm[dofs[3]]
        v_j_c = u_comm[dofs[4]]
        th_j_c = u_comm[dofs[5]]

        q_trial = np.array([cx * (u_j - u_i) + cy * (v_j - v_i), th_j - th_i], dtype=float)
        q_comm = np.array([cx * (u_j_c - u_i_c) + cy * (v_j_c - v_i_c), th_j_c - th_i_c], dtype=float)
        dq = q_trial - q_comm

        info = self.hinge.hinge.update(dq, commit=False)
        s_vec = info["s"]
        Kt = info["Kt"]

        B = np.array(
            [
                [-cx, -cy, 0.0, cx, cy, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        k_l = B.T @ Kt @ B
        f_l = B.T @ s_vec

        self._trial = {"s_new": s_vec.copy(), "q_p_new": info["q_p"].copy()}
        out = {"dq": dq.copy(), "N": float(s_vec[0]), "M": float(s_vec[1]), "active": info["active"]}
        return k_l, f_l, out

    def commit(self) -> None:
        if self._trial is None:
            return
        self.hinge.hinge.s = self._trial["s_new"].copy()
        self.hinge.hinge.q_p = self._trial["q_p_new"].copy()
        self._trial = None

    def reset_state(self) -> None:
        self._trial = None
        self.hinge.hinge.s = np.zeros(2, float)
        self.hinge.hinge.q_p = np.zeros(2, float)
