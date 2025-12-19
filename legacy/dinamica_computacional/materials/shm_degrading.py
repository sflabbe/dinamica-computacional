from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SHMParams:
    K0_0: float
    My_0: float
    alpha_post: float = 0.02
    cK: float = 2.0
    cMy: float = 1.0
