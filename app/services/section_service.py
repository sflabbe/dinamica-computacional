from __future__ import annotations

from dc_solver.sections import AluminumRectTube, load_steel_profile, list_steel_profiles

from .session_schema import SectionSelection


def available_section_families() -> dict[str, list[str]]:
    return {"steel": ["IPE", "HEA", "HEB"], "aluminum": ["rect_tube"]}


def available_profiles(material: str, family: str | None = None) -> list[str]:
    mat = material.lower().strip()
    if mat == "steel":
        if family is None:
            names: list[str] = []
            for fam in available_section_families()["steel"]:
                names.extend(list_steel_profiles(fam))
            return sorted(names)
        return list_steel_profiles(family)
    if mat == "aluminum":
        return ["AL-RECT-TUBE"]
    raise ValueError(f"Unsupported material: {material}")


def build_section(selection: SectionSelection):
    mat = selection.material.lower().strip()
    if mat == "steel":
        if selection.family is None or selection.name is None:
            raise ValueError("Steel selection requires 'family' and 'name'.")
        return load_steel_profile(selection.family, selection.name)

    if mat == "aluminum":
        p = selection.params
        return AluminumRectTube(
            name=str(selection.name or "AL-RECT-TUBE"),
            b=float(p.get("b", 0.2)),
            h=float(p.get("h", 0.3)),
            t=float(p.get("t", 0.01)),
            E=float(p.get("E", 70e9)),
            fy=float(p.get("fy", 250e6)),
        )
    raise ValueError(f"Unsupported material: {selection.material}")


def section_properties_table(selection: SectionSelection) -> dict[str, float | str]:
    sec = build_section(selection)
    props = sec.properties()
    return {
        "name": props.name,
        "A": props.A,
        "I_y": props.I_y,
        "I_z": props.I_z,
        "W_el_y": props.W_el_y,
        "W_pl_y": props.W_pl_y,
        "W_el_z": props.W_el_z,
        "W_pl_z": props.W_pl_z,
        "E": props.E,
        "fy": props.fy,
        "source": props.source,
    }
