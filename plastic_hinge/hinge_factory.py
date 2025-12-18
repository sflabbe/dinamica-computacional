from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .rc_section import RCSectionRect
from .nm_surface import NMSurfacePolygon
from .hinge_nm import PlasticHingeNM

def hinge_from_section(section: RCSectionRect,
                       Lp: float,
                       interaction_points: Optional[np.ndarray] = None,
                       KN: Optional[float] = None,
                       KM: Optional[float] = None) -> PlasticHingeNM:
    """Build a PlasticHingeNM from an RC section.

    - interaction_points: optional precomputed points (N,M). If None, it will sample them.
    - KN, KM: optional stiffnesses. If None, we provide simple defaults for prototyping.

    Default stiffnesses are *not* code-calibrated; replace with your preferred elastic extraction (EA/Lp, EI/Lp, etc.).
    """
    if interaction_points is None:
        interaction_points = section.sample_interaction_curve()

    surface = NMSurfacePolygon.from_points(interaction_points)

    if KN is None:
        # very rough axial stiffness proxy
        Ec_eff = 0.35 * section.fc / max(section.eps_c0, 1e-9)  # secant-like
        EA = Ec_eff * section.Ac
        KN = EA / max(Lp, 1e-9)

    if KM is None:
        # very rough rotational stiffness proxy: EI/Lp with I=b*h^3/12 and Ec_eff as above
        Ec_eff = 0.35 * section.fc / max(section.eps_c0, 1e-9)
        I = section.b * section.h**3 / 12.0
        EI = Ec_eff * I
        KM = EI / max(Lp, 1e-9)

    K = np.diag([KN, KM])
    return PlasticHingeNM(surface=surface, K=K)
