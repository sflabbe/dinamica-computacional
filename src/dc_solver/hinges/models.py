"""Hinge constitutive models and rotational spring element."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

import numpy as np

from plastic_hinge import NMSurfacePolygon, PlasticHingeNM, FiberSection2DStateful

if TYPE_CHECKING:
    from dc_solver.fem.nodes import Node

# Import JIT kernel for SHM hinge evaluation (guarded by DC_FAST)
try:
    from dc_solver.kernels.hinge_jit import shm_bouc_wen_step, is_jit_enabled
except ImportError:
    # Fallback if kernels module not available
    shm_bouc_wen_step = None
    is_jit_enabled = lambda: False


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
    """Smooth Hysteretic Model (SHM) beam-end rotational hinge (educational).

    This implementation is inspired by Sivaselvan & Reinhorn (2000), using a
    Bouc–Wen style internal variable `z` (dimensionless) and an optional
    axial-moment interaction for *beam* hinges via a reduced My under axial
    compression:

        My_eff = My_0 * (1 - N_comp/N_cr)^eta,  with floor My_eff >= My_floor_frac_N * My_0

    Notes
    -----
    * The model is tuned for robust Newton iterations (non-negative tangent).
    * Degradation uses a dimensionless work measure `a` and simple linear/exponential
      softening coefficients (b1,b2,cK,cMy). Defaults are mild.
    """

    K0_0: float
    My_0: float

    # Post-yield stiffness ratio (Kpost/K0)
    alpha_post: float = 0.03

    # Bouc–Wen parameters
    bw_A: float = 0.0   # <=0 enables auto-scaling A_eff = K0/My
    bw_beta: float = 0.6
    bw_gamma: float = 0.4
    bw_n: float = 10.0

    # Optional pinching (disabled by default for monotonic/gravity consistency)
    pinch: float = 0.0
    theta_pinch: float = 0.002

    # Degradation controls (dimensionless work a)
    b1: float = 0.05    # stiffness degradation vs a
    b2: float = 0.15    # strength degradation vs a
    cK: float = 0.0     # exponential stiffness degradation vs a
    cMy: float = 0.0    # exponential strength degradation vs a
    E_ref_mult: float = 20.0
    K0_min_frac: float = 0.05
    My_min_frac: float = 0.05

    # Axial interaction (beam hinges): N_comp is compression-positive
    eta: float = 1.0
    N_cr: float = 0.0
    My_floor_frac_N: float = 0.6
    N_comp_current: float = 0.0

    # Increment integration controls
    dth_sub_max: float = 2e-4
    max_substeps: int = 400

    # Committed state
    th_comm: float = 0.0
    z_comm: float = 0.0
    a_comm: float = 0.0
    M_comm: float = 0.0

    def _My0_base(self) -> float:
        """Base (pre-degradation) My, including optional My(N) reduction for compression."""
        My0 = float(self.My_0)
        Ncr = float(self.N_cr)
        if Ncr > 0.0:
            Ncomp = max(0.0, float(self.N_comp_current))
            ratio = min(Ncomp / max(Ncr, 1e-12), 0.999)
            My_eff = My0 * (1.0 - ratio) ** float(self.eta)
            My_eff = max(float(self.My_floor_frac_N) * My0, float(My_eff))
            return float(My_eff)
        return float(My0)

    def _eref(self) -> float:
        """Reference work to keep `a` dimensionless and stable."""
        K0 = float(max(self.K0_0, 1e-18))
        My0 = float(max(abs(self._My0_base()), 1e-18))
        theta_y = My0 / K0
        return float(max(My0 * theta_y, 1e-12))

    def _energy_ref(self) -> float:
        K0 = float(max(self.K0_0, 1e-18))
        My0 = float(max(abs(self._My0_base()), 1e-18))
        theta_y = My0 / K0
        return float(max(1e-12, float(self.E_ref_mult) * My0 * max(theta_y, 1e-12)))

    def _bw_rhs(self, z: float, sign_dth: float, A_eff: float) -> float:
        return float(A_eff) - float(self.bw_beta) * sign_dth * abs(z) ** (float(self.bw_n) - 1.0) * z - float(self.bw_gamma) * abs(z) ** float(self.bw_n)

    def _degraded_K0(self, a: float) -> float:
        fac_lin = max(float(self.K0_min_frac), 1.0 - float(self.b1) * float(a))
        fac_exp = math.exp(-float(self.cK) * float(a)) if float(self.cK) != 0.0 else 1.0
        return float(max(float(self.K0_min_frac) * float(self.K0_0), float(self.K0_0) * fac_lin * fac_exp))

    def _degraded_My(self, a: float) -> float:
        My0 = abs(float(self._My0_base()))
        fac_lin = max(float(self.My_min_frac), 1.0 - float(self.b2) * float(a))
        fac_exp = math.exp(-float(self.cMy) * float(a)) if float(self.cMy) != 0.0 else 1.0
        return float(max(float(self.My_min_frac) * My0, My0 * fac_lin * fac_exp))

    def eval_increment(self, dth: float, nsub: int | None = None) -> Tuple[float, float, float, float, float, float]:
        # Use JIT kernel if available (DC_FAST=1 and numba installed)
        if shm_bouc_wen_step is not None:
            # Gather parameters for JIT kernel
            My0_abs = max(abs(float(self._My0_base())), 1e-12)
            Eref_dimless = float(self._eref())

            # Ignore nsub parameter when using JIT (kernel determines substeps internally)
            return shm_bouc_wen_step(
                dth=float(dth),
                th_comm=float(self.th_comm),
                z_comm=float(self.z_comm),
                a_comm=float(self.a_comm),
                M_comm=float(self.M_comm),
                K0_0=float(self.K0_0),
                My0_abs=My0_abs,
                alpha_post=float(self.alpha_post),
                bw_A=float(self.bw_A),
                bw_beta=float(self.bw_beta),
                bw_gamma=float(self.bw_gamma),
                bw_n=float(self.bw_n),
                b1=float(self.b1),
                b2=float(self.b2),
                cK=float(self.cK),
                cMy=float(self.cMy),
                K0_min_frac=float(self.K0_min_frac),
                My_min_frac=float(self.My_min_frac),
                Eref_dimless=Eref_dimless,
                pinch=float(self.pinch),
                theta_pinch=float(self.theta_pinch),
                dth_sub_max=float(self.dth_sub_max),
                max_substeps=int(self.max_substeps),
            )

        # Fallback: original Python implementation
        if nsub is None:
            dmax = float(max(self.dth_sub_max, 1e-12))
            nsub = max(1, int(np.ceil(abs(float(dth)) / dmax)))
        nsub = int(min(max(int(nsub), 1), int(max(self.max_substeps, 1))))
        dth_sub = float(dth) / float(nsub)

        th = float(self.th_comm)
        z = float(self.z_comm)
        a = float(self.a_comm)
        M = float(self.M_comm)

        Eref_dimless = float(self._eref())

        # Auto-scale Bouc–Wen A so that initial tangent ≈ K0.
        My0_abs = max(abs(float(self._My0_base())), 1e-12)
        if float(self.bw_A) > 0.0:
            A_eff = float(self.bw_A)
        else:
            A_eff = float(max(float(self.K0_0) / My0_abs, 1e-6))

        for _ in range(int(nsub)):
            th_new = th + dth_sub
            sign_dth = 1.0 if dth_sub >= 0.0 else -1.0

            def rhs(z_local: float) -> float:
                return self._bw_rhs(z_local, sign_dth, A_eff)

            # RK4 integration for z(θ)
            k1 = rhs(z)
            k2 = rhs(z + 0.5 * dth_sub * k1)
            k3 = rhs(z + 0.5 * dth_sub * k2)
            k4 = rhs(z + dth_sub * k3)
            z_new = z + (dth_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Degraded properties
            K0 = self._degraded_K0(a)
            My = self._degraded_My(a)

            # Pinching (only for reloading through origin)
            pinch_factor = 1.0
            if float(self.pinch) > 0.0 and (th_new * sign_dth) < 0.0:
                pinch_factor = 1.0 - float(self.pinch) * math.exp(-abs(th_new) / max(float(self.theta_pinch), 1e-12))

            M_new = float(self.alpha_post) * float(K0) * float(th_new) + (1.0 - float(self.alpha_post)) * float(My) * float(z_new) * float(pinch_factor)

            # accumulate non-dimensional hysteretic work (always positive)
            dW = 0.5 * (float(M) + float(M_new)) * float(dth_sub)
            a += abs(float(dW)) / float(Eref_dimless)

            th = float(th_new)
            z = float(max(-1.2, min(1.2, float(z_new))))
            M = float(M_new)

        # Tangent stiffness (keep non-negative for robustness)
        K0 = self._degraded_K0(a)
        My = self._degraded_My(a)
        sign_tot = 1.0 if float(dth) >= 0.0 else -1.0
        dz_dth = float(self._bw_rhs(float(z), sign_tot, A_eff))
        dz_dth = max(0.0, dz_dth)
        pinch_factor_end = 1.0
        if float(self.pinch) > 0.0 and (float(th) * sign_tot) < 0.0:
            pinch_factor_end = 1.0 - float(self.pinch) * math.exp(-abs(float(th)) / max(float(self.theta_pinch), 1e-12))

        k_tan = float(self.alpha_post) * float(K0) + (1.0 - float(self.alpha_post)) * float(My) * float(dz_dth) * float(pinch_factor_end)
        k_tan = float(max(k_tan, max(float(self.alpha_post) * float(K0), 1e-12)))

        return float(M), float(k_tan), float(th), float(z), float(a), float(M)


class FiberBeamHinge1D:
    """1D beam-end rotational hinge backed by a 2D fiber section (stateful steel).

    Notes
    -----
    * The hinge kinematics is expressed via a *plastic rotation* variable `th_pl`.
    * We map hinge rotation to curvature via:

        kappa = kappa_factor * th_pl / Lp

      where `kappa_factor=2` corresponds to a triangular curvature distribution
      over the plastic hinge length Lp.

    * The fiber section uses y measured **downwards** from the top fiber.
      With the usual structural sign convention (positive moment -> bottom tension),
      this means the raw section bending moment can have the opposite sign.
      `moment_sign=-1` flips the sign so that M aligns with +th.

    * Axial equilibrium is enforced by solving for eps0 such that N(eps0,kappa)=N_target.
    """

    kind = "beam_fiber"

    def __init__(
        self,
        *,
        section: FiberSection2DStateful,
        Lp: float,
        N_target: float = 0.0,
        kappa_factor: float = 2.0,
        moment_sign: float = -1.0,
        # eps0 solve
        tol_N: float = 1e3,
        max_iter_eps0: int = 50,
        eps0_bracket: float = 5e-3,
        deps_max: float = 5e-4,
        # global robustness
        line_search: bool = False,
        ls_max: int = 12,
        # tangent regularization
        k_floor_ratio: float = 1e-4,
        k_floor_abs: float = 1.0,
    ):
        self.section = section
        self.Lp = float(Lp)
        self.N_target = float(N_target)
        self.kappa_factor = float(kappa_factor)
        self.moment_sign = float(moment_sign)

        self.tol_N = float(tol_N)
        self.max_iter_eps0 = int(max_iter_eps0)
        self.eps0_bracket = float(eps0_bracket)
        self.deps_max = float(deps_max)

        self.line_search = bool(line_search)
        self.ls_max = int(ls_max)

        self.k_floor_ratio = float(k_floor_ratio)
        self.k_floor_abs = float(k_floor_abs)

        # committed state
        self.th_pl_comm = 0.0
        self.a_comm = 0.0
        self.eps0_comm = 0.0

        # trial state
        self.th_pl_trial = 0.0
        self.a_trial = 0.0
        self.eps0_trial = 0.0

        # last eval outputs
        self.M = 0.0
        self.k_tan = 0.0

    def reset_state(self) -> None:
        self.th_pl_comm = 0.0
        self.a_comm = 0.0
        self.eps0_comm = 0.0
        self.th_pl_trial = 0.0
        self.a_trial = 0.0
        self.eps0_trial = 0.0
        self.M = 0.0
        self.k_tan = 0.0
        self.section.reset_state()

    def _eval_N_tan(self, eps0: float, kappa: float):
        N, M, dN_de0, dN_dk, dM_de0, dM_dk = self.section.response_tangent(eps0, kappa)
        return float(N), float(dN_de0), float(dN_dk), float(M), float(dM_de0), float(dM_dk)

    def _solve_eps0(self, kappa: float, eps0_init: float) -> float:
        """Solve eps0 for axial equilibrium N(eps0,kappa)=N_target.

        Uses a damped Newton with optional backtracking. If Newton cannot make
        progress (near-zero tangent), falls back to a bracketed bisection.
        """
        kappa = float(kappa)
        eps0 = float(eps0_init)
        best_eps0 = eps0
        best_abs = float('inf')

        # Newton iterations
        for _it in range(self.max_iter_eps0):
            N, dN_de0, _dN_dk, _M, _dM_de0, _dM_dk = self._eval_N_tan(eps0, kappa)
            res = N - self.N_target
            ares = abs(res)
            if ares < best_abs:
                best_abs = ares
                best_eps0 = eps0
            if ares <= self.tol_N:
                return eps0
            if abs(dN_de0) < 1e-12:
                break

            deps = -res / dN_de0
            # clamp for robustness
            deps = float(max(-self.deps_max, min(self.deps_max, deps)))

            if not self.line_search:
                eps0 = eps0 + deps
                continue

            # backtracking on |res|
            res0 = ares
            alpha = 1.0
            accepted = False
            for _ls in range(self.ls_max):
                eps_try = eps0 + alpha * deps
                N_try, dN_de0_try, *_ = self._eval_N_tan(eps_try, kappa)
                res_try = N_try - self.N_target
                if abs(res_try) <= (1.0 - 1e-4 * alpha) * res0:
                    eps0 = eps_try
                    accepted = True
                    break
                # If tangent vanishes, accept the smallest step to avoid stalling
                if abs(dN_de0_try) < 1e-12:
                    alpha *= 0.5
                else:
                    alpha *= 0.5
            if not accepted:
                eps0 = eps0 + 0.25 * deps

        # Fallback: bracket + bisection around the best Newton iterate
        center = float(best_eps0)
        half = float(self.eps0_bracket)

        def f(x: float) -> float:
            N, *_ = self._eval_N_tan(x, kappa)
            return float(N - self.N_target)

        a = center - half
        b = center + half
        fa = f(a)
        fb = f(b)
        # widen bracket if needed
        for _ in range(10):
            if fa == 0.0:
                return a
            if fb == 0.0:
                return b
            if fa * fb < 0.0:
                break
            half *= 2.0
            a = center - half
            b = center + half
            fa = f(a)
            fb = f(b)
        else:
            # no sign change found -> return best Newton value
            return float(best_eps0)

        # bisection
        lo, hi = a, b
        flo, fhi = fa, fb
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) <= self.tol_N:
                return mid
            if flo * fmid < 0.0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi)

    def eval_increment(self, u_trial: np.ndarray, u_comm: np.ndarray, dofs: Tuple[int, int]) -> Tuple[float, float, Dict]:
        """Return moment M and tangent k for the *trial* step."""
        dof_i, dof_j = int(dofs[0]), int(dofs[1])
        th_comm = float(u_comm[dof_j] - u_comm[dof_i])
        th_new = float(u_trial[dof_j] - u_trial[dof_i])
        dth = th_new - th_comm

        # Hinge plastic rotation state (here: total hinge rotation)
        self.th_pl_trial = float(self.th_pl_comm + dth)
        kappa = self.kappa_factor * self.th_pl_trial / max(self.Lp, 1e-12)

        # Solve for eps0 that satisfies axial target (does not update state)
        eps0 = self._solve_eps0(kappa, self.eps0_comm)
        self.eps0_trial = float(eps0)

        # Single trial update to compute tangents and update *trial* steel state
        N, M_sec, dN_de0, dN_dk, dM_de0, dM_dk = self.section.trial_update(eps0, kappa)

        # Eliminate eps0 using axial constraint: dM/dk|N
        if abs(dN_de0) > 1e-12:
            dM_dk_eff = float(dM_dk - dM_de0 * (dN_dk / dN_de0))
        else:
            dM_dk_eff = float(dM_dk)

        # Map to hinge moment sign convention and to dM/dtheta
        M = self.moment_sign * float(M_sec)
        k_theta = self.moment_sign * float(dM_dk_eff) * (self.kappa_factor / max(self.Lp, 1e-12))

        # Regularize tangent (avoid a singular global stiffness)
        k_ref = max(self.k_floor_abs, self.k_floor_ratio * abs(float(dM_dk_eff) * (self.kappa_factor / max(self.Lp, 1e-12))))
        if not np.isfinite(k_theta) or k_theta <= 0.0:
            k_theta = float(k_ref)

        self.M = float(M)
        self.k_tan = float(k_theta)

        # dissipated-energy proxy for future extensions / reporting
        self.a_trial = float(self.a_comm + abs(M) * abs(dth))

        info = {
            "kind": self.kind,
            "M": float(M),
            "k": float(k_theta),
            "th": float(th_new),
            "dth": float(dth),
            "th_pl": float(self.th_pl_trial),
            "kappa": float(kappa),
            "eps0": float(eps0),
            "N": float(N),
            "N_target": float(self.N_target),
            "resN": float(N - self.N_target),
        }
        return self.M, self.k_tan, info

    def commit_trial(self) -> None:
        self.th_pl_comm = float(self.th_pl_trial)
        self.a_comm = float(self.a_trial)
        self.eps0_comm = float(self.eps0_trial)
        self.section.commit_trial()


@dataclass
class FiberRotSpringElement:
    """Zero-length rotational spring element backed by a FiberBeamHinge1D."""

    ni: int
    nj: int
    hinge: FiberBeamHinge1D
    nodes: List[Node]
    kind: str = "beam_fiber"

    # Optional coupling: provide an axial force target (N_target) to the fiber hinge
    # from an associated frame element axial force.
    #
    # Convention:
    # - FrameElementLinear2D reports N in *tension-positive* convention.
    # - Fiber sections in this repo use *compression-positive* convention.
    # Therefore, a typical choice is beam_sign = -1.0.
    beam_idx: int | None = None
    beam_sign: float = -1.0
    name: str = ""

    # Updated by Model.assemble when beam_idx is set.
    _beam_N_tension: float = 0.0

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

        # Fiber hinge uses only the relative rotation between the two end nodes.
        M, kM, info = self.hinge.eval_increment(u_trial, u_comm, (int(dofs[2]), int(dofs[5])))

        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = float(kM) * (Bm.T @ Bm)
        f_l = (Bm.T * float(M)).reshape(6)

        # store for commit
        self._trial = {"ok": True}

        # provide a consistent info payload for plotting/debugging
        info_out = {"dtheta": float(info.get("dth", info.get("dtheta", 0.0))), "M": float(M)}
        info_out.update(info)

        # Optional debug/trace metadata
        if self.name:
            info_out["name"] = str(self.name)
        if self.beam_idx is not None:
            info_out["beam_idx"] = int(self.beam_idx)
            info_out["N_beam_tension"] = float(self._beam_N_tension)
            info_out["N_target_used"] = float(self.hinge.N_target)
        return k_l, f_l, info_out

    def commit(self) -> None:
        if self._trial is None:
            return
        self.hinge.commit_trial()
        self._trial = None

    def reset_state(self) -> None:
        self._trial = None
        self.hinge.reset_state()
@dataclass
class RotSpringElement:
    """Zero-length rotational spring element.

    Supported kinds
    ---------------
    * "col_nm"    : ColumnHingeNMRot (My depends on N via Model.update_column_yields)
    * "beam_shm"  : SHMBeamHinge1D (optionally with My(N) via N_comp_current)
    * "beam_fiber": FiberBeamHinge1D (compat mode; prefer FiberRotSpringElement)
    """

    ni: int
    nj: int
    kind: str

    col_hinge: Optional[ColumnHingeNMRot] = None
    beam_hinge: Optional[object] = None  # SHMBeamHinge1D or (compat) FiberBeamHinge1D
    nodes: List[Node] = None  # injected by builder

    # Optional: provide a dedicated fiber hinge object (compat mode)
    fiber_hinge: Optional["FiberBeamHinge1D"] = None

    # Optional coupling to a frame element axial force (tension-positive convention)
    beam_idx: int | None = None
    beam_sign: float = -1.0  # multiply N_tension -> N_comp (compression-positive)
    name: str = ""

    # Updated during Model.assemble when beam_idx is set
    _beam_N_tension: float = 0.0

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

    def _fiber_obj(self):
        return self.fiber_hinge if self.fiber_hinge is not None else self.beam_hinge

    def eval_trial(self, u_trial: np.ndarray, u_comm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        dofs = self.dofs()

        # Relative rotation between end nodes (theta_j - theta_i)
        dtheta = float(u_trial[int(dofs[5])] - u_trial[int(dofs[2])])
        dtheta_comm = float(u_comm[int(dofs[5])] - u_comm[int(dofs[2])])
        dth_inc = dtheta - dtheta_comm

        kind = str(self.kind).lower().strip()

        if kind == "col_nm":
            if self.col_hinge is None:
                raise ValueError("col_nm hinge requested but col_hinge is None")
            M, kM, th_p_new, a_new, M_new = self.col_hinge.eval_increment(dth_inc)
            z_new = None
            dW_pl = abs(float(M_new) * float(dth_inc))
            self._trial = {"kind": kind, "th_p_new": float(th_p_new), "a_new": float(a_new), "M_new": float(M_new)}

            info = {
                "kind": "col_nm",
                "name": str(self.name) if self.name else "",
                "dtheta": float(dth_inc),
                "M": float(M_new),
                "k": float(kM),
                "a": float(a_new),
                "th_p": float(th_p_new),
                "dW_pl": float(dW_pl),
            }

        elif kind == "beam_shm":
            if self.beam_hinge is None:
                raise ValueError("beam_shm hinge requested but beam_hinge is None")
            # SHM hinge works with rotation increments
            M_new, kM, th_new, z_new, a_new, _ = self.beam_hinge.eval_increment(dth_inc)
            dW_pl = abs(0.5 * (float(getattr(self.beam_hinge, "M_comm", 0.0)) + float(M_new)) * float(dth_inc))
            self._trial = {"kind": kind, "th_new": float(th_new), "z_new": float(z_new), "a_new": float(a_new), "M_new": float(M_new)}

            # Optional My(N) reporting
            My_base = None
            util = None
            N_comp = None
            N_cr = None
            try:
                if hasattr(self.beam_hinge, "_My0_base"):
                    My_base = float(abs(self.beam_hinge._My0_base()))
                elif hasattr(self.beam_hinge, "My_0"):
                    My_base = float(abs(getattr(self.beam_hinge, "My_0")))
                if My_base is not None and My_base > 0:
                    util = float(abs(float(M_new)) / max(My_base, 1e-12))
                if hasattr(self.beam_hinge, "N_comp_current"):
                    N_comp = float(getattr(self.beam_hinge, "N_comp_current"))
                if hasattr(self.beam_hinge, "N_cr"):
                    N_cr = float(getattr(self.beam_hinge, "N_cr"))
            except Exception:
                pass

            info = {
                "kind": "beam_shm",
                "name": str(self.name) if self.name else "",
                "dtheta": float(dth_inc),
                "M": float(M_new),
                "k": float(kM),
                "th": float(th_new),
                "z": float(z_new),
                "a": float(a_new),
                "dW_pl": float(dW_pl),
            }
            if My_base is not None:
                info["My_eff"] = float(My_base)
            if util is not None:
                info["util_My"] = float(util)
            if N_comp is not None:
                info["N_comp"] = float(N_comp)
            if N_cr is not None:
                info["N_cr"] = float(N_cr)

        elif kind in ("beam_fiber", "fiber"):
            fh = self._fiber_obj()
            if fh is None:
                raise ValueError("beam_fiber hinge requested but no fiber hinge object was provided")
            # Fiber hinge needs the full vectors and DOF indices.
            M_new, kM, info_f = fh.eval_increment(u_trial, u_comm, (int(dofs[2]), int(dofs[5])))
            z_new = None
            a_new = float(getattr(fh, "a_trial", getattr(fh, "a_comm", 0.0)))
            dW_pl = abs(float(M_new) * float(dth_inc))
            self._trial = {"kind": "beam_fiber"}

            # Merge the fiber hinge info payload (includes N, eps0, etc.)
            info = {"kind": "beam_fiber", "name": str(self.name) if self.name else "", "dtheta": float(dth_inc), "M": float(M_new), "k": float(kM), "a": float(a_new), "dW_pl": float(dW_pl)}
            if isinstance(info_f, dict):
                info.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info_f.items()})

        else:
            raise ValueError(f"Unknown hinge kind: {self.kind!r} (name={self.name!r})")

        # Local element stiffness/force (rotation-only)
        Bm = np.array([[0, 0, -1, 0, 0, 1]], float)
        k_l = float(kM) * (Bm.T @ Bm)
        f_l = (Bm.T * float(M_new)).reshape(6)
        return k_l, f_l, info

    def commit(self) -> None:
        if self._trial is None:
            return
        kind = str(self._trial.get("kind", self.kind)).lower().strip()

        if kind == "col_nm" and self.col_hinge is not None:
            self.col_hinge.th_p_comm = float(self._trial.get("th_p_new", self.col_hinge.th_p_comm))
            self.col_hinge.a_comm = float(self._trial.get("a_new", self.col_hinge.a_comm))
            self.col_hinge.M_comm = float(self._trial.get("M_new", self.col_hinge.M_comm))

        elif kind == "beam_shm" and self.beam_hinge is not None:
            self.beam_hinge.th_comm = float(self._trial.get("th_new", getattr(self.beam_hinge, "th_comm", 0.0)))
            self.beam_hinge.z_comm = float(self._trial.get("z_new", getattr(self.beam_hinge, "z_comm", 0.0)))
            self.beam_hinge.a_comm = float(self._trial.get("a_new", getattr(self.beam_hinge, "a_comm", 0.0)))
            self.beam_hinge.M_comm = float(self._trial.get("M_new", getattr(self.beam_hinge, "M_comm", 0.0)))

        elif kind in ("beam_fiber", "fiber"):
            fh = self._fiber_obj()
            if fh is not None and hasattr(fh, "commit_trial"):
                fh.commit_trial()

        self._trial = None

    def reset_state(self) -> None:
        self._trial = None
        kind = str(self.kind).lower().strip()

        if kind == "col_nm" and self.col_hinge is not None:
            self.col_hinge.th_p_comm = 0.0
            self.col_hinge.a_comm = 0.0
            self.col_hinge.M_comm = 0.0

        elif kind == "beam_shm" and self.beam_hinge is not None:
            self.beam_hinge.th_comm = 0.0
            self.beam_hinge.z_comm = 0.0
            self.beam_hinge.a_comm = 0.0
            self.beam_hinge.M_comm = 0.0

        elif kind in ("beam_fiber", "fiber"):
            fh = self._fiber_obj()
            if fh is not None and hasattr(fh, "reset_state"):
                fh.reset_state()

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