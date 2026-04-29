from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SectionSelection:
    material: str
    family: str | None = None
    name: str | None = None
    params: dict[str, float | str] = field(default_factory=dict)


@dataclass
class FrameInput:
    width: float
    height: float
    n_col: int = 4
    n_beam: int = 6
    section: SectionSelection | None = None
    mass_total: float = 0.0
    damping_ratio: float = 0.05


@dataclass
class AnalysisSettings:
    run_gravity: bool = True
    run_modal: bool = True
    run_dynamic: bool = False
    n_modes: int = 6
    integrator: str = "hht"
