"""Hinge constitutive models and rotational spring element."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from plastic_hinge import NMSurfacePolygon, PlasticHingeNM, FiberSection2DStateful
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
class FiberBeamHinge1D:
    """Beam end hinge based on a stateful fiber section (RC).

    The hinge uses a plastic-hinge length Lp and assumes:
        theta = kappa * Lp  =>  kappa = theta / Lp

    For the *beam hinge* we typically want **pure bending**, so we enforce:
        N(eps0, kappa) = N_target
    by solving for eps0 each evaluation.

    Notes
    -----
    * Steel is elastic-perfect plastic with plastic-strain memory (handled by
      FiberSection2DStateful).
    * Concrete is compression-only (parabolic-rectangular) and path-independent.
    """

    section: FiberSection2DStateful
    Lp: float
    N_target: float = 0.0
    max_iter: int = 25
    tol_N: float = 1e3
    tol_eps0: float = 1e-10

    # committed state
    th_comm: float = 0.0
    M_comm: float = 0.0
    a_comm: float = 0.0
    eps0_comm: float = 0.0

    # last trial cache
    _trial: Optional[Dict] = None

    def reset_state(self) -> None:
        self.th_comm = 0.0
        self.M_comm = 0.0
        self.a_comm = 0.0
        self.eps0_comm = 0.0
        self._trial = None
        if hasattr(self.section, "reset_state"):
            self.section.reset_state()

    def _solve_eps0_newton(self, kappa: float) -> Tuple[float, int, float]:
        """Solve N(eps0, kappa) = N_target for eps0 using Newton with fallback."""

        eps0 = float(self.eps0_comm)
        it = 0
        resN = 0.0
        for it in range(1, int(self.max_iter) + 1):
            N, _M, dN_de0, _dN_dk, _dM_de0, _dM_dk = self.section.response_tangent(eps0, kappa)
            resN = float(N - self.N_target)
            if abs(resN) <= self.tol_N:
                return eps0, it, resN
            if abs(dN_de0) < 1e-16:
                break
            deps = -resN / float(dN_de0)
            # simple damping to avoid wild jumps in deep plasticity
            deps = float(np.clip(deps, -2e-3, 2e-3))
            eps0_new = eps0 + deps
            if abs(eps0_new - eps0) <= self.tol_eps0:
                eps0 = eps0_new
                return eps0, it, resN
            eps0 = eps0_new

        # Fallback: bracket + bisection around last eps0
        a = eps0 - 0.01
        b = eps0 + 0.01
        Na, *_ = self.section.response_tangent(a, kappa)
        Nb, *_ = self.section.response_tangent(b, kappa)
        fa = float(Na - self.N_target)
        fb = float(Nb - self.N_target)
        # expand bracket if needed
        for _ in range(10):
            if fa * fb <= 0.0:
                break
            a -= 0.01
            b += 0.01
            Na, *_ = self.section.response_tangent(a, kappa)
            Nb, *_ = self.section.response_tangent(b, kappa)
            fa = float(Na - self.N_target)
            fb = float(Nb - self.N_target)

        if fa * fb > 0.0:
            # no bracket: return last Newton eps0
            return eps0, it, resN

        for j in range(30):
            m = 0.5 * (a + b)
            Nm, *_ = self.section.response_tangent(m, kappa)
            fm = float(Nm - self.N_target)
            if abs(fm) <= self.tol_N:
                return m, it + j + 1, fm
            if fa * fm <= 0.0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
        m = 0.5 * (a + b)
        Nm, *_ = self.section.response_tangent(m, kappa)
        fm = float(Nm - self.N_target)
        return m, it + 30, fm

    def eval_increment(self, dth: float) -> Tuple[float, float, float, float, float, Dict]:
        """Evaluate increment dtheta and store trial state.

        Returns
        -------
        M, k_tan, th_new, a_new, M_new, extra
        """
        th_new = float(self.th_comm) + float(dth)
        kappa = th_new / max(float(self.Lp), 1e-12)

        eps0, iters, resN = self._solve_eps0_newton(kappa)

        # Final evaluation and update steel plastic strains to a *trial* state
        N, M, dN_de0, dN_dk, dM_de0, dM_dk = self.section.trial_update(eps0, kappa)

        # Effective tangent under constraint dN = 0: de0/dk = -dN_dk/dN_de0
        if abs(dN_de0) > 1e-16:
            de0_dk = -float(dN_dk) / float(dN_de0)
            dM_dk_eff = float(dM_dk) + float(dM_de0) * de0_dk
        else:
            dM_dk_eff = float(dM_dk)
        k_tan = dM_dk_eff / max(float(self.Lp), 1e-12)

        # cumulative dissipation proxy
        dW = 0.5 * (float(self.M_comm) + float(M)) * float(dth)
        a_new = float(self.a_comm) + abs(float(dW))

        self._trial = {
            "th_new": th_new,
            "M_new": float(M),
            "a_new": a_new,
            "eps0_new": float(eps0),
            "kappa": float(kappa),
            "N_res": float(N - self.N_target),
            "iters": int(iters),
        }

        extra = {
            "N": float(N),
            "N_res": float(N - self.N_target),
            "eps0": float(eps0),
            "kappa": float(kappa),
            "iters": int(iters),
        }
        return float(M), float(k_tan), float(th_new), float(a_new), float(M), extra

    def commit(self) -> None:
        if self._trial is None:
            return
        # commit steel trial state first
        if hasattr(self.section, "commit_trial"):
            self.section.commit_trial()
        self.th_comm = float(self._trial["th_new"])
        self.M_comm = float(self._trial["M_new"])
        self.a_comm = float(self._trial["a_new"])
        self.eps0_comm = float(self._trial["eps0_new"])
        self._trial = None


@dataclass
class FiberRotSpringElement:
    """Zero-length rotational spring element backed by a FiberBeamHinge1D."""

    ni: int
    nj: int
    hinge: FiberBeamHinge1D
    nodes: List[Node]
    kind: str = "beam_fiber"

    _trial: Optional[Dict] = None

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

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        dofs = self.dofs()
        th_i = float(u_trial[dofs[2]])
        th_j = float(u_trial[dofs[5]])
        th_i_c = float(u_comm[dofs[2]])
        th_j_c = float(u_comm[dofs[5]])
        dth_inc = (th_j - th_i) - (th_j_c - th_i_c)

        M, kM, th_new, a_new, M_new, extra = self.hinge.eval_increment(dth_inc)

        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = float(kM) * (Bm.T @ Bm)
        f_l = (Bm.T * float(M)).reshape(6)

        self._trial = {"th_new": th_new, "a_new": a_new, "M_new": M_new}
        info = {"dtheta": float(dth_inc), "M": float(M), "a": float(a_new)}
        info.update(extra)
        return k_l, f_l, info

    def commit(self) -> None:
        self.hinge.commit()
        self._trial = None

    def reset_state(self) -> None:
        self._trial = None
        self.hinge.reset_state()


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
