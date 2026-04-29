from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SectionProperties:
    """Section properties with strict SI units.

    Units:
    - A [m^2]
    - I_* [m^4]
    - W_* [m^3]
    - i_* [m]
    - E, fy [Pa]
    """

    name: str
    A: float
    I_y: float
    I_z: float
    W_el_y: float
    W_pl_y: float
    W_el_z: float
    W_pl_z: float
    i_y: float
    i_z: float
    E: float
    fy: float
    source: str
    notes: str = ""

    @property
    def EA(self) -> float:
        return self.E * self.A

    @property
    def EI_y(self) -> float:
        return self.E * self.I_y

    @property
    def EI_z(self) -> float:
        return self.E * self.I_z

    @property
    def M_el_y(self) -> float:
        return self.W_el_y * self.fy

    @property
    def M_pl_y(self) -> float:
        return self.W_pl_y * self.fy

    @property
    def N_pl(self) -> float:
        return self.A * self.fy


class SectionLike(Protocol):
    def properties(self) -> SectionProperties:
        ...
