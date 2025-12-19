"""Linear elastic material definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ElasticMaterial:
    name: str
    E: float
    density: float = 0.0
