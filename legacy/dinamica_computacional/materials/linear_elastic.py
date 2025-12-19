from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearElastic:
    E: float
    nu: float = 0.3
