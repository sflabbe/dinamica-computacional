from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class RebarLayer:
    """A rebar layer at depth y from top fiber (y=0 at top)."""
    As: float   # steel area [m^2]
    y: float    # depth from top [m]

@dataclass
class RCSectionRect:
    """Rectangular RC section (b x h) with discrete rebar layers.

    Simple fiber section:
    - Concrete: parabolic-rectangular compression block
    - Concrete tension neglected
    - Steel: bilinear elastic-perfectly plastic (±fy)

    Units: consistent SI (m, N, Pa) recommended.
    """
    b: float
    h: float
    fc: float      # concrete compressive strength [Pa]
    fy: float      # steel yield stress [Pa]
    Es: float = 200e9
    eps_c0: float = 0.002
    eps_cu: float = 0.003
    layers: List[RebarLayer] = None
    n_fibers: int = 200

    def __post_init__(self):
        if self.layers is None:
            self.layers = []

    @property
    def Ac(self) -> float:
        return self.b * self.h

    @property
    def y_c(self) -> float:
        return 0.5 * self.h

    def _sigma_c(self, eps: np.ndarray) -> np.ndarray:
        """Concrete stress (compression positive)."""
        eps = np.asarray(eps, float)
        sig = np.zeros_like(eps)
        # compression only
        comp = eps > 0
        e = eps[comp]
        # parabolic up to eps_c0, then flat up to eps_cu
        r = np.clip(e / self.eps_c0, 0.0, 1.0)
        sig_par = self.fc * (2*r - r*r)
        sig_flat = np.full_like(e, self.fc)
        sig_c = np.where(e <= self.eps_c0, sig_par, sig_flat)
        sig[comp] = np.where(e <= self.eps_cu, sig_c, self.fc)
        return sig

    def _sigma_s(self, eps: np.ndarray) -> np.ndarray:
        eps = np.asarray(eps, float)
        sig = self.Es * eps
        sig = np.clip(sig, -self.fy, self.fy)
        return sig

    def response(self, eps0: float, kappa: float) -> Tuple[float, float]:
        """Return (N,M) for a plane-sections strain field:
            eps(y) = eps0 + kappa*(y_c - y)
        with y measured from top. Compression positive.
        M is about centroid (positive if compression at top gives positive M).
        """
        # Concrete fibers along depth
        y = np.linspace(0.0, self.h, self.n_fibers)
        dy = self.h / (self.n_fibers - 1)
        # fiber areas (strip)
        A_f = self.b * dy
        eps_f = eps0 + kappa*(self.y_c - y)
        sig_c = self._sigma_c(eps_f)
        N_c = float(np.sum(sig_c * A_f))
        M_c = float(np.sum(sig_c * A_f * (y - self.y_c)))  # lever arm about centroid

        # Steel layers
        N_s = 0.0
        M_s = 0.0
        for lay in self.layers:
            eps_s = eps0 + kappa*(self.y_c - lay.y)
            sig_s = float(self._sigma_s(eps_s))
            N_s += sig_s * lay.As
            M_s += sig_s * lay.As * (lay.y - self.y_c)

        N = N_c + N_s
        M = M_c + M_s
        return N, M

    def sample_interaction_curve(self, n: int = 80, c_min: float = 1e-4, c_max: Optional[float]=None) -> np.ndarray:
        """Build a crude N–M interaction curve by scanning neutral axis depth c.

        Assumption: extreme top strain fixed at eps_cu, curvature = eps_cu / c.
        For each c:
            eps(y) = kappa*(c - y)  (zero at y=c)
        Convert to (eps0, kappa) form used by response().

        This gives a reasonable *envelope* for teaching / prototyping.
        """
        if c_max is None:
            c_max = 5.0 * self.h

        c_vals = np.geomspace(c_min, c_max, n)
        pts = []
        for c in c_vals:
            kappa = self.eps_cu / c
            # eps(y)=kappa*(c - y) ; rewrite eps0 + kappa*(y_c - y)
            # => eps0 = kappa*(c - y_c)
            eps0 = kappa * (c - self.y_c)
            N, M = self.response(eps0, kappa)
            pts.append([N, M])

        # Add pure tension (steel only) and pure compression (approx)
        As_tot = sum(l.As for l in self.layers)
        pts.append([-As_tot * self.fy, 0.0])

        # crude pure compression: take kappa=0 and eps0=eps_cu
        N0, M0 = self.response(self.eps_cu, 0.0)
        pts.append([N0, 0.0])

        return np.asarray(pts, float)
