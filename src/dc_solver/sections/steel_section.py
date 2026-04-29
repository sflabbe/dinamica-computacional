from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dc_solver.sections.base import SectionProperties


@dataclass(frozen=True)
class SteelISection:
    name: str
    h: float
    b: float
    tw: float
    tf: float
    A: float
    I_y: float
    I_z: float
    W_el_y: float
    W_pl_y: float
    W_el_z: float
    W_pl_z: float
    i_y: float
    i_z: float
    E: float = 210e9
    fy: float = 355e6
    source: str = ""
    notes: str = ""

    def properties(self) -> SectionProperties:
        return SectionProperties(
            name=self.name,
            A=self.A,
            I_y=self.I_y,
            I_z=self.I_z,
            W_el_y=self.W_el_y,
            W_pl_y=self.W_pl_y,
            W_el_z=self.W_el_z,
            W_pl_z=self.W_pl_z,
            i_y=self.i_y,
            i_z=self.i_z,
            E=self.E,
            fy=self.fy,
            source=self.source,
            notes=self.notes,
        )

    def elastic_utilization_my(self, M_ed: float) -> float:
        return abs(M_ed) / self.properties().M_el_y

    def plastic_utilization_my(self, M_ed: float) -> float:
        return abs(M_ed) / self.properties().M_pl_y

    def axial_utilization(self, N_ed: float) -> float:
        return abs(N_ed) / self.properties().N_pl

    def preclassify_ec3_flexure(self) -> dict:
        """Preliminary geometric classification helper inspired by EN 1993-1-1 limits.
        It is not a full EC3 design check and does not replace a prüffähiger Nachweis.
        """
        return {
            "status": "manual_verification_required",
            "reason": "normative limits not encoded in Phase 3",
        }


_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
_PROFILE_FILES = {"IPE": "ipe.json", "HEA": "hea.json", "HEB": "heb.json"}


def _cm2_to_m2(v: float) -> float:
    return v * 1e-4


def _cm3_to_m3(v: float) -> float:
    return v * 1e-6


def _cm4_to_m4(v: float) -> float:
    return v * 1e-8


def _mm_to_m(v: float) -> float:
    return v * 1e-3


def _profile_record_to_section(record: dict, *, E: float, fy: float) -> SteelISection:
    return SteelISection(
        name=record["name"],
        h=_mm_to_m(record["h_mm"]),
        b=_mm_to_m(record["b_mm"]),
        tw=_mm_to_m(record["tw_mm"]),
        tf=_mm_to_m(record["tf_mm"]),
        A=_cm2_to_m2(record["A_cm2"]),
        I_y=_cm4_to_m4(record["Iy_cm4"]),
        I_z=_cm4_to_m4(record["Iz_cm4"]),
        W_el_y=_cm3_to_m3(record["Wel_y_cm3"]),
        W_pl_y=_cm3_to_m3(record["Wpl_y_cm3"]),
        W_el_z=_cm3_to_m3(record["Wel_z_cm3"]),
        W_pl_z=_cm3_to_m3(record["Wpl_z_cm3"]),
        i_y=record["iy_cm"] * 1e-2,
        i_z=record["iz_cm"] * 1e-2,
        E=E,
        fy=fy,
        source=record.get("source", ""),
        notes=f"units_original={record.get('units_original', 'unknown')}; review_required={record.get('review_required', True)}",
    )


def _load_series(series: str) -> list[dict]:
    key = series.upper()
    if key not in _PROFILE_FILES:
        raise ValueError(f"Unknown series '{series}'. Expected one of: {sorted(_PROFILE_FILES)}")
    path = _PROFILE_DIR / _PROFILE_FILES[key]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_steel_profiles(series: str | None = None) -> list[str]:
    if series is None:
        names: list[str] = []
        for s in _PROFILE_FILES:
            names.extend([r["name"] for r in _load_series(s)])
        return sorted(names)
    return [r["name"] for r in _load_series(series)]


def load_steel_profile(series: str, name: str, *, E: float = 210e9, fy: float = 355e6) -> SteelISection:
    for record in _load_series(series):
        if record["name"] == name:
            return _profile_record_to_section(record, E=E, fy=fy)
    raise ValueError(f"Profile '{name}' not found in series '{series}'.")
