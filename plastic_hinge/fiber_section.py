from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Tuple
import numpy as np

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
        sig_par = self.fc * (2*r - r*r)
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

@dataclass
class FiberSection:
    """Generic fiber section. Plane sections remain plane.

    Strain field:
        eps(y) = eps0 + kappa*(y_c - y)
    with y from top, y_c centroid reference (user-defined).
    """
    fibers: List[Fiber]
    y_c: float

    def response(self, eps0: float, kappa: float) -> Tuple[float, float]:
        N = 0.0
        M = 0.0
        for f in self.fibers:
            eps = eps0 + kappa*(self.y_c - f.y)
            sig = float(f.mat.stress(np.array([eps]))[0])
            N += sig * f.A
            M += sig * f.A * (f.y - self.y_c)
        return float(N), float(M)


# --- 2D fiber mesh (y-z) ----------------------------------------------------


@dataclass(frozen=True)
class Fiber2D:
    """2D fiber with centroid (y,z).

    Notes
    -----
    In the current codebase we mainly use *uniaxial* bending (N–M envelope)
    where the plane-sections strain field depends only on y. A 2D mesh is still
    useful to:

    - match common "fiber section" definitions in structural software/papers
    - support future extension to biaxial bending (N–Mx–My)
    """

    A: float
    y: float
    z: float
    mat: Material


@dataclass
class FiberSection2D:
    """2D fiber section. Plane sections remain plane.

    Uniaxial strain field used here:
        eps(y) = eps0 + kappa*(y_c - y)

    (i.e., curvature about the z-axis). For N–M envelopes this is sufficient.
    """

    fibers: List[Fiber2D]
    y_c: float
    z_c: float = 0.0

    def response(self, eps0: float, kappa: float) -> Tuple[float, float]:
        N = 0.0
        M = 0.0
        for f in self.fibers:
            eps = eps0 + kappa * (self.y_c - f.y)
            sig = float(f.mat.stress(np.array([eps]))[0])
            N += sig * f.A
            M += sig * f.A * (f.y - self.y_c)
        return float(N), float(M)


def _clustered_edges_01(n: int, kind: str = "cosine") -> np.ndarray:
    """Return n+1 edges in [0,1] with optional clustering near 0 and 1.

    kind="cosine" uses a Chebyshev/cosine mapping which clusters near the
    boundaries. This is a cheap, robust alternative to explicit medial-axis
    methods for simple convex shapes.
    """

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
    """Fast 2D fiber mesh for a rectangular section.

    Parameters
    ----------
    b, h : float
        Width (z-direction) and height (y-direction) [m].
    ny, nz : int
        Number of cells in y and z.
    mat : Material
        Material assigned to all fibers.
    clustering : {"uniform","cosine"}
        Edge distribution. "cosine" clusters fibers near the boundary (cheap
        distance-to-boundary refinement, often similar to what you'd get from
        medial-axis driven sizing in convex sections).
    """

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
