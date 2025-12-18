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
