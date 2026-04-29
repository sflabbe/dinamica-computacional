from app.services.section_service import (
    available_profiles,
    available_section_families,
    build_section,
    section_properties_table,
)
from app.services.session_schema import SectionSelection


def test_services_importable_without_streamlit():
    families = available_section_families()
    assert "steel" in families


def test_build_section_returns_properties():
    name = available_profiles("steel", "IPE")[0]
    sec = build_section(SectionSelection(material="steel", family="IPE", name=name))
    props = sec.properties()
    assert props.A > 0.0
    table = section_properties_table(SectionSelection(material="steel", family="IPE", name=name))
    assert float(table["I_y"]) > 0.0
