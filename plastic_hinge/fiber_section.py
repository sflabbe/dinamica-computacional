from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple, Optional, Dict, Any

import numpy as np

from ._numba import njit, USE_NUMBA


class Material(Protocol):
    def stress(self, eps: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Fiber:
    A: float   # area [m^2]
    y: float   # depth from top [m]
    mat: Material


@dataclass
class ConcreteParabolicRect:
    fc: float
    eps_c0: float = 0.002
    eps_cu: float = 0.003

    def stress(self, eps: np.ndarray) -> np.ndarray:
        eps = np.asarray(eps, float)
        sig = np.zeros_like(eps)
        comp = eps > 0
        e = eps[comp]
        r = np.clip(e / self.eps_c0, 0.0, 1.0)
        sig_par = self.fc * (2 * r - r * r)
        sig_flat = np.full_like(e, self.fc)
        sig_c = np.where(e <= self.eps_c0, sig_par, sig_flat)
        sig[comp] = np.where(e <= self.eps_cu, sig_c, self.fc)
        return sig


@dataclass
class SteelBilinearPerfect:
    fy: float
    Es: float = 200e9

    def stress(self, eps: np.ndarray) -> np.ndarray:
        eps = np.asarray(eps, float)
        sig = self.Es * eps
        return np.clip(sig, -self.fy, self.fy)


# --- Numba kernels (optional) ------------------------------------------------

@njit(cache=True)
def _sigma_c_scalar(eps: float, fc: float, eps_c0: float, eps_cu: float) -> float:
    # compression positive; tension neglected
    if eps <= 0.0:
        return 0.0
    if eps <= eps_c0:
        r = eps / eps_c0
        return fc * (2.0 * r - r * r)
    if eps <= eps_cu:
        return fc
    return fc


@njit(cache=True)
def _sigma_s_scalar(eps: float, fy: float, Es: float) -> float:
    sig = Es * eps
    if sig > fy:
        return fy
    if sig < -fy:
        return -fy
    return sig


@njit(cache=True)
def _fiber_response_kernel(
    eps0: float,
    kappa: float,
    A: np.ndarray,
    y: np.ndarray,
    mat_id: np.ndarray,
    y_c: float,
    fc: float,
    eps_c0: float,
    eps_cu: float,
    fy: float,
    Es: float,
) -> Tuple[float, float]:
    N = 0.0
    M = 0.0
    n = A.shape[0]
    for i in range(n):
        eps = eps0 + kappa * (y_c - y[i])
        if mat_id[i] == 0:
            sig = _sigma_c_scalar(eps, fc, eps_c0, eps_cu)
        else:
            sig = _sigma_s_scalar(eps, fy, Es)
        N += sig * A[i]
        M += sig * A[i] * (y[i] - y_c)
    return N, M


@dataclass
class FiberSection:
    """Generic fiber section. Plane sections remain plane.

    Strain field:
        eps(y) = eps0 + kappa*(y_c - y)
    with y from top, y_c centroid reference (user-defined).
    """

    fibers: List[Fiber]
    y_c: float

    # internal cache for numba-packed arrays (created lazily)
    _pack: Optional[Dict[str, Any]] = None

    def _pack_for_numba(self) -> Optional[Dict[str, Any]]:
        """Pack fibers into arrays for the JIT kernel.

        Supported materials:
        - ConcreteParabolicRect
        - SteelBilinearPerfect

        If other materials are present, returns None (falls back to pure Python).
        """
        n = len(self.fibers)
        if n == 0:
            return None

        A = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        mat_id = np.empty(n, dtype=np.int8)

        conc: Optional[ConcreteParabolicRect] = None
        steel: Optional[SteelBilinearPerfect] = None

        for i, f in enumerate(self.fibers):
            A[i] = float(f.A)
            y[i] = float(f.y)
            if isinstance(f.mat, ConcreteParabolicRect):
                mat_id[i] = 0
                conc = f.mat if conc is None else conc
            elif isinstance(f.mat, SteelBilinearPerfect):
                mat_id[i] = 1
                steel = f.mat if steel is None else steel
            else:
                return None

        if conc is None:
            conc = ConcreteParabolicRect(fc=0.0)
        if steel is None:
            steel = SteelBilinearPerfect(fy=0.0)

        return {
            "A": A,
            "y": y,
            "mat_id": mat_id,
            "fc": float(conc.fc),
            "eps_c0": float(conc.eps_c0),
            "eps_cu": float(conc.eps_cu),
            "fy": float(steel.fy),
            "Es": float(steel.Es),
        }

    def response(self, eps0: float, kappa: float) -> Tuple[float, float]:
        if USE_NUMBA:
            if self._pack is None:
                self._pack = self._pack_for_numba()
            if self._pack is not None:
                p = self._pack
                return tuple(
                    float(v)
                    for v in _fiber_response_kernel(
                        float(eps0),
                        float(kappa),
                        p["A"],
                        p["y"],
                        p["mat_id"],
                        float(self.y_c),
                        float(p["fc"]),
                        float(p["eps_c0"]),
                        float(p["eps_cu"]),
                        float(p["fy"]),
                        float(p["Es"]),
                    )
                )
        # fallback (generic)
        N = 0.0
        M = 0.0
        for f in self.fibers:
            eps = float(eps0 + kappa * (self.y_c - f.y))
            sig = float(f.mat.stress(np.array([eps], dtype=float))[0])
            N += sig * f.A
            M += sig * f.A * (f.y - self.y_c)
        return float(N), float(M)


# --- 2D fiber mesh (y-z) ----------------------------------------------------


@dataclass(frozen=True)
class Fiber2D:
    """2D fiber with centroid (y,z)."""

    A: float
    y: float
    z: float
    mat: Material


@dataclass
class FiberSection2D:
    """2D fiber section. Plane sections remain plane (uniaxial curvature here)."""

    fibers: List[Fiber2D]
    y_c: float
    z_c: float = 0.0

    _pack: Optional[Dict[str, Any]] = None

    def _pack_for_numba(self) -> Optional[Dict[str, Any]]:
        n = len(self.fibers)
        if n == 0:
            return None

        A = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        mat_id = np.empty(n, dtype=np.int8)

        conc: Optional[ConcreteParabolicRect] = None
        steel: Optional[SteelBilinearPerfect] = None

        for i, f in enumerate(self.fibers):
            A[i] = float(f.A)
            y[i] = float(f.y)
            if isinstance(f.mat, ConcreteParabolicRect):
                mat_id[i] = 0
                conc = f.mat if conc is None else conc
            elif isinstance(f.mat, SteelBilinearPerfect):
                mat_id[i] = 1
                steel = f.mat if steel is None else steel
            else:
                return None

        if conc is None:
            conc = ConcreteParabolicRect(fc=0.0)
        if steel is None:
            steel = SteelBilinearPerfect(fy=0.0)

        return {
            "A": A,
            "y": y,
            "mat_id": mat_id,
            "fc": float(conc.fc),
            "eps_c0": float(conc.eps_c0),
            "eps_cu": float(conc.eps_cu),
            "fy": float(steel.fy),
            "Es": float(steel.Es),
        }

    def response(self, eps0: float, kappa: float) -> Tuple[float, float]:
        if USE_NUMBA:
            if self._pack is None:
                self._pack = self._pack_for_numba()
            if self._pack is not None:
                p = self._pack
                return tuple(
                    float(v)
                    for v in _fiber_response_kernel(
                        float(eps0),
                        float(kappa),
                        p["A"],
                        p["y"],
                        p["mat_id"],
                        float(self.y_c),
                        float(p["fc"]),
                        float(p["eps_c0"]),
                        float(p["eps_cu"]),
                        float(p["fy"]),
                        float(p["Es"]),
                    )
                )
        # fallback (generic)
        N = 0.0
        M = 0.0
        for f in self.fibers:
            eps = float(eps0 + kappa * (self.y_c - f.y))
            sig = float(f.mat.stress(np.array([eps], dtype=float))[0])
            N += sig * f.A
            M += sig * f.A * (f.y - self.y_c)
        return float(N), float(M)


# --- Stateful steel (elastic-perfect plastic) for cyclic fiber hinges ---------


@njit(cache=True)
def _sigma_c_ceb90_tangent_scalar(eps: float, fc: float, eps_c0: float, E_ci: float) -> Tuple[float, float]:
    """CEB-90 uniaxial concrete: ascending branch + post-peak softening.

    Compression-positive convention. Tension neglected.

    Parameters
    ----------
    eps   : total compressive strain (compression > 0)
    fc    : peak compressive stress [Pa]
    eps_c0: strain at peak stress [-]
    E_ci  : initial tangent modulus [Pa]  (E_ci = 21500e6*(fc_MPa/10)^(1/3))

    Returns
    -------
    sig : stress [Pa]
    Et  : tangent modulus [Pa]  (< 0 on softening branch)

    Reference
    ---------
    CEB-FIP Model Code 1990, implemented following compression.py in the
    cdp-generator project (CEB-90 Sargin formula with smooth post-peak).
    """
    if eps <= 0.0:
        return 0.0, 0.0

    E_c1 = fc / eps_c0                      # secant modulus at peak
    eta_E = E_ci / E_c1                      # tangent/secant ratio (≈ 2.0–2.5 for NSC)
    r = eps / eps_c0

    # --- inflection point (boundary ascending / descending) ---
    tmp = 0.5 * eta_E + 1.0
    disc = 0.25 * tmp * tmp - 0.5
    if disc < 0.0:
        disc = 0.0
    eta_lim = 0.5 * tmp + disc ** 0.5       # = e_clim / eps_c0

    if r <= eta_lim:
        # ---- ascending branch (Sargin formula) ----
        denom = 1.0 + (eta_E - 2.0) * r
        if denom <= 1e-30:
            return fc, 0.0
        sig = fc * (eta_E * r - r * r) / denom
        # analytical tangent: dσ/dε = fc*(η-2r-(η-2)r²) / (denom²·eps_c0)
        num_Et = eta_E - 2.0 * r - (eta_E - 2.0) * r * r
        Et = fc * num_Et / (denom * denom * eps_c0)
        sig = max(0.0, min(sig, fc))
        return sig, Et
    else:
        # ---- descending / softening branch ----
        xi_num = 4.0 * (eta_lim * eta_lim * (eta_E - 2.0) + 2.0 * eta_lim - eta_E)
        xi_den = (eta_lim * (eta_E - 2.0) + 1.0) ** 2
        xi = xi_num / xi_den if xi_den > 1e-30 else 0.0

        a_coef = xi / eta_lim - 2.0 / (eta_lim * eta_lim)
        b_coef = 4.0 / eta_lim - xi
        denom_d = a_coef * r * r + b_coef * r
        if denom_d <= 1e-30:
            return 0.0, 0.0
        sig = fc / denom_d
        sig = max(0.0, min(sig, fc))
        # tangent (negative → softening)
        Et = -fc * (2.0 * a_coef * r + b_coef) / (denom_d * denom_d * eps_c0)
        return sig, Et


# Keep old name as alias for backward compatibility with non-stateful kernel
@njit(cache=True)
def _sigma_c_tangent_scalar(eps: float, fc: float, eps_c0: float, eps_cu: float) -> Tuple[float, float]:
    """Legacy parabolic-rect model. Used only by non-stateful FiberSection."""
    if eps <= 0.0:
        return 0.0, 0.0
    if eps <= eps_c0:
        r = eps / eps_c0
        sig = fc * (2.0 * r - r * r)
        Et = (2.0 * fc / eps_c0) * (1.0 - r)
        return sig, Et
    if eps <= eps_cu:
        return fc, 0.0
    return fc, 0.0


@njit(cache=True)
def _sigma_s_ep_tangent_scalar(eps: float, eps_p_old: float, fy: float, Es: float) -> Tuple[float, float, float]:
    """Steel elastic-perfect plastic return mapping (1D) with plastic strain state."""
    sig_trial = Es * (eps - eps_p_old)
    f = abs(sig_trial) - fy
    if f <= 0.0:
        return sig_trial, Es, eps_p_old
    sgn = 1.0 if sig_trial >= 0.0 else -1.0
    sig = fy * sgn
    eps_p_new = eps - sig / Es
    return sig, 0.0, eps_p_new


@njit(cache=True)
def _fiber_response_tangent_stateful_kernel(
    eps0: float,
    kappa: float,
    A: np.ndarray,
    y: np.ndarray,
    mat_id: np.ndarray,
    eps_p_old: np.ndarray,
    y_c: float,
    fc: float,
    eps_c0: float,
    eps_cu: float,
    fy: float,
    Es: float,
    do_update: bool,
    eps_p_new: np.ndarray,
    E_ci: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    """Return N,M and tangents dN/de0, dN/dk, dM/de0, dM/dk.

    If do_update=True, eps_p_new is written (steel fibers updated). Otherwise, eps_p_new
    is ignored (may be used as scratch).

    E_ci: initial tangent modulus for CEB-90 concrete model [Pa].
    If E_ci > 0, uses CEB-90 (with post-peak softening).
    If E_ci <= 0, falls back to legacy parabolic-rect model.
    """
    N = 0.0
    M = 0.0
    dN_de0 = 0.0
    dN_dk = 0.0
    dM_de0 = 0.0
    dM_dk = 0.0

    n = A.shape[0]
    for i in range(n):
        eps = eps0 + kappa * (y_c - y[i])
        if mat_id[i] == 0:
            if E_ci > 0.0:
                sig, Et = _sigma_c_ceb90_tangent_scalar(eps, fc, eps_c0, E_ci)
            else:
                sig, Et = _sigma_c_tangent_scalar(eps, fc, eps_c0, eps_cu)
            if do_update:
                eps_p_new[i] = eps_p_old[i]
        else:
            sig, Et, epn = _sigma_s_ep_tangent_scalar(eps, eps_p_old[i], fy, Es)
            if do_update:
                eps_p_new[i] = epn

        Ai = A[i]
        lever = (y[i] - y_c)
        d_eps_dk = (y_c - y[i])

        N += sig * Ai
        M += sig * Ai * lever

        dN_de0 += Et * Ai
        dN_dk += Et * Ai * d_eps_dk
        dM_de0 += Et * Ai * lever
        dM_dk += Et * Ai * lever * d_eps_dk

    return N, M, dN_de0, dN_dk, dM_de0, dM_dk


@dataclass
class FiberSection2DStateful:
    """2D fiber section with stateful steel (plastic strain per fiber).

    Notes
    -----
    * Supports only ConcreteParabolicRect (mat_id=0) and SteelBilinearPerfect (mat_id=1)
      for packing/JIT.
    * Uses an **elastic-perfect plastic** steel model with plastic-strain memory.
    """

    fibers: List[Fiber2D]
    y_c: float
    z_c: float = 0.0

    _pack: Optional[Dict[str, Any]] = None
    _eps_p: Optional[np.ndarray] = None
    _eps_p_trial: Optional[np.ndarray] = None

    def _pack_for_numba(self) -> Optional[Dict[str, Any]]:
        n = len(self.fibers)
        if n == 0:
            return None

        A = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        mat_id = np.empty(n, dtype=np.int8)

        conc: Optional[ConcreteParabolicRect] = None
        steel: Optional[SteelBilinearPerfect] = None

        for i, f in enumerate(self.fibers):
            A[i] = float(f.A)
            y[i] = float(f.y)
            if isinstance(f.mat, ConcreteParabolicRect):
                mat_id[i] = 0
                conc = f.mat if conc is None else conc
            elif isinstance(f.mat, SteelBilinearPerfect):
                mat_id[i] = 1
                steel = f.mat if steel is None else steel
            else:
                return None

        if conc is None:
            conc = ConcreteParabolicRect(fc=0.0)
        if steel is None:
            steel = SteelBilinearPerfect(fy=0.0)

        # CEB-90 initial tangent modulus: E_ci = 21 500 MPa * (f_cm/10 MPa)^(1/3)
        # fc is in Pa here; convert to MPa, apply formula, convert back.
        fc_MPa = float(conc.fc) / 1.0e6
        E_ci_Pa = 21500.0e6 * (fc_MPa / 10.0) ** (1.0 / 3.0) if fc_MPa > 0.0 else 0.0

        return {
            "A": A,
            "y": y,
            "mat_id": mat_id,
            "fc": float(conc.fc),
            "eps_c0": float(conc.eps_c0),
            "eps_cu": float(conc.eps_cu),
            "fy": float(steel.fy),
            "Es": float(steel.Es),
            "E_ci": float(E_ci_Pa),
        }

    def reset_state(self) -> None:
        """Reset steel plastic strains to zero."""
        if self._eps_p is not None:
            self._eps_p[:] = 0.0

    def _ensure_state(self) -> None:
        if self._pack is None:
            self._pack = self._pack_for_numba()
        if self._pack is None:
            raise RuntimeError("FiberSection2DStateful: unsupported materials for JIT/packing")
        n = int(self._pack["A"].shape[0])
        if self._eps_p is None or self._eps_p.shape[0] != n:
            self._eps_p = np.zeros(n, dtype=float)
        if self._eps_p_trial is None or self._eps_p_trial.shape[0] != n:
            self._eps_p_trial = np.zeros(n, dtype=float)

    def response_tangent(self, eps0: float, kappa: float) -> Tuple[float, float, float, float, float, float]:
        """Return N,M and tangents without updating state (trial evaluation)."""
        self._ensure_state()
        assert self._pack is not None and self._eps_p is not None and self._eps_p_trial is not None
        p = self._pack
        N, M, dN_de0, dN_dk, dM_de0, dM_dk = _fiber_response_tangent_stateful_kernel(
            float(eps0),
            float(kappa),
            p["A"],
            p["y"],
            p["mat_id"],
            self._eps_p,
            float(self.y_c),
            float(p["fc"]),
            float(p["eps_c0"]),
            float(p["eps_cu"]),
            float(p["fy"]),
            float(p["Es"]),
            False,
            self._eps_p_trial,
            float(p.get("E_ci", 0.0)),
        )
        return float(N), float(M), float(dN_de0), float(dN_dk), float(dM_de0), float(dM_dk)

    def trial_update(self, eps0: float, kappa: float) -> Tuple[float, float, float, float, float, float]:
        """Return N,M and tangents and update *trial* steel plastic strains.

        This does **not** modify the committed plastic strains. Call commit_trial()
        after the global step is accepted.
        """
        self._ensure_state()
        assert self._pack is not None and self._eps_p is not None and self._eps_p_trial is not None
        p = self._pack
        N, M, dN_de0, dN_dk, dM_de0, dM_dk = _fiber_response_tangent_stateful_kernel(
            float(eps0),
            float(kappa),
            p["A"],
            p["y"],
            p["mat_id"],
            self._eps_p,
            float(self.y_c),
            float(p["fc"]),
            float(p["eps_c0"]),
            float(p["eps_cu"]),
            float(p["fy"]),
            float(p["Es"]),
            True,
            self._eps_p_trial,
            float(p.get("E_ci", 0.0)),
        )

        return float(N), float(M), float(dN_de0), float(dN_dk), float(dM_de0), float(dM_dk)

    def commit_trial(self) -> None:
        """Commit trial plastic strains (after an accepted global increment)."""
        self._ensure_state()
        assert self._eps_p is not None and self._eps_p_trial is not None
        self._eps_p[:] = self._eps_p_trial


def _clustered_edges_01(n: int, kind: str = "cosine") -> np.ndarray:
    """Return n+1 edges in [0,1] with optional clustering near 0 and 1."""

    u = np.linspace(0.0, 1.0, int(n) + 1)
    if kind == "uniform":
        return u
    if kind == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * u))
    raise ValueError(f"Unknown clustering kind: {kind!r}")


def rectangular_fiber_mesh(
    *,
    b: float,
    h: float,
    ny: int,
    nz: int,
    mat: Material,
    clustering: str = "cosine",
) -> List[Fiber2D]:
    """Fast 2D fiber mesh for a rectangular section."""

    y_edges = h * _clustered_edges_01(ny, kind=clustering)
    z_edges = (-0.5 * b) + b * _clustered_edges_01(nz, kind=clustering)

    fibers: List[Fiber2D] = []
    for iy in range(ny):
        y0, y1 = float(y_edges[iy]), float(y_edges[iy + 1])
        y_mid = 0.5 * (y0 + y1)
        dy = y1 - y0
        for iz in range(nz):
            z0, z1 = float(z_edges[iz]), float(z_edges[iz + 1])
            z_mid = 0.5 * (z0 + z1)
            dz = z1 - z0
            fibers.append(Fiber2D(A=dy * dz, y=y_mid, z=z_mid, mat=mat))
    return fibers