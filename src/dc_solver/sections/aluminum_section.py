from __future__ import annotations

from dataclasses import dataclass

from dc_solver.sections.base import SectionProperties


@dataclass(frozen=True)
class AluminumRectTube:
    name: str
    b: float
    h: float
    t: float
    E: float = 70e9
    fy: float = 250e6
    source: str = "user-defined rectangular tube"

    def properties(self) -> SectionProperties:
        b_i = self.b - 2.0 * self.t
        h_i = self.h - 2.0 * self.t
        A = self.b * self.h - b_i * h_i
        I_y = self.b * self.h**3 / 12.0 - b_i * h_i**3 / 12.0
        I_z = self.h * self.b**3 / 12.0 - h_i * b_i**3 / 12.0
        W_el_y = I_y / (self.h / 2.0)
        W_el_z = I_z / (self.b / 2.0)
        # simple placeholder for section helper: use elastic modulus for plastic slot
        W_pl_y = W_el_y
        W_pl_z = W_el_z
        i_y = (I_y / A) ** 0.5
        i_z = (I_z / A) ** 0.5
        return SectionProperties(
            name=self.name,
            A=A,
            I_y=I_y,
            I_z=I_z,
            W_el_y=W_el_y,
            W_pl_y=W_pl_y,
            W_el_z=W_el_z,
            W_pl_z=W_pl_z,
            i_y=i_y,
            i_z=i_z,
            E=self.E,
            fy=self.fy,
            source=self.source,
            notes="section property helper",
        )

    def preclassify_ec9_placeholder(self) -> dict:
        return {
            "status": "manual_verification_required",
            "reason": "EC9 classification is not implemented in Phase 3",
        }
