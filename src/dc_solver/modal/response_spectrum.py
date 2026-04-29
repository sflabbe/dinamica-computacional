from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOIL_PARAMS = {
    "A": dict(S=1.0, TB=0.15, TC=0.40, TD=2.0),
    "B": dict(S=1.2, TB=0.15, TC=0.50, TD=2.0),
    "C": dict(S=1.15, TB=0.20, TC=0.60, TD=2.0),
    "D": dict(S=1.35, TB=0.20, TC=0.80, TD=2.0),
    "E": dict(S=1.4, TB=0.15, TC=0.50, TD=2.0),
}


@dataclass
class SpectrumEC8:
    """Preliminary elastic response spectrum helper, not a jurisdiction-ready EC8/NA design module."""

    ag: float
    gamma_I: float = 1.0
    eta: float = 1.0
    soil: str = "C"

    def Sa(self, T):
        p = SOIL_PARAMS[self.soil]
        S, TB, TC, TD = p["S"], p["TB"], p["TC"], p["TD"]
        T_arr = np.asarray(T, dtype=float)
        Te = np.maximum(T_arr, 1e-9)
        out = np.empty_like(Te)
        c = self.ag * self.gamma_I * S * self.eta
        out[Te <= TB] = c * (1.0 + (Te[Te <= TB] / TB) * 1.5)
        mask2 = (Te > TB) & (Te <= TC)
        out[mask2] = c * 2.5
        mask3 = (Te > TC) & (Te <= TD)
        out[mask3] = c * 2.5 * (TC / Te[mask3])
        mask4 = Te > TD
        out[mask4] = c * 2.5 * (TC * TD / (Te[mask4] ** 2))
        return out if np.ndim(T) else float(out[0])


@dataclass
class SpectralResult:
    combined: float
    method: str


def spectral_combination(values, *, method="srss") -> SpectralResult:
    vals = np.asarray(values, dtype=float)
    if method.lower() == "srss":
        return SpectralResult(combined=float(np.sqrt(np.sum(vals**2))), method="srss")
    if method.lower() == "cqc":
        raise NotImplementedError("CQC combination is not implemented in Phase 2.")
    raise ValueError("Unknown combination method.")
