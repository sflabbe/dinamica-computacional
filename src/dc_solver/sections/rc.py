from __future__ import annotations

from plastic_hinge.rc_section import RCSectionRect, RebarLayer

from dc_solver.sections.base import SectionProperties


def rc_gross_properties_rect(name: str, b: float, h: float, E_cm: float, fc: float) -> SectionProperties:
    """Gross uncracked rectangle properties (no RC design checks)."""
    A = b * h
    I_y = b * h**3 / 12.0
    I_z = h * b**3 / 12.0
    return SectionProperties(
        name=name,
        A=A,
        I_y=I_y,
        I_z=I_z,
        W_el_y=I_y / (h / 2.0),
        W_pl_y=b * h**2 / 4.0,
        W_el_z=I_z / (b / 2.0),
        W_pl_z=h * b**2 / 4.0,
        i_y=(I_y / A) ** 0.5,
        i_z=(I_z / A) ** 0.5,
        E=E_cm,
        fy=fc,
        source="gross rectangle formula",
        notes="uncracked gross section property helper",
    )
