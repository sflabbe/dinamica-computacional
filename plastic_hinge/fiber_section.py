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
