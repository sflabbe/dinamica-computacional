from __future__ import annotations

import streamlit as st

from app.services import (
    SectionSelection,
    available_profiles,
    section_properties_table,
)


def render_section_selector(key_prefix: str = "section") -> SectionSelection:
    material = st.selectbox(
        "Material",
        options=["steel", "aluminum"],
        key=f"{key_prefix}_material",
    )

    if material == "steel":
        family = st.selectbox("Steel family", ["IPE", "HEA", "HEB"], key=f"{key_prefix}_family")
        profile_names = available_profiles(material, family)
        name = st.selectbox("Profile", profile_names, key=f"{key_prefix}_name")
        return SectionSelection(material=material, family=family, name=name, params={})

    default_params = {
        "b": st.number_input("b [m]", min_value=0.01, value=0.20, step=0.01, key=f"{key_prefix}_b"),
        "h": st.number_input("h [m]", min_value=0.01, value=0.30, step=0.01, key=f"{key_prefix}_h"),
        "t": st.number_input("t [m]", min_value=0.001, value=0.01, step=0.001, key=f"{key_prefix}_t"),
        "E": st.number_input("E [Pa]", min_value=1.0, value=70e9, step=1e9, key=f"{key_prefix}_E"),
        "fy": st.number_input("fy [Pa]", min_value=1.0, value=250e6, step=1e6, key=f"{key_prefix}_fy"),
    }
    return SectionSelection(material=material, family="rect_tube", name="AL-RECT-TUBE", params=default_params)


def render_section_properties(selection: SectionSelection) -> None:
    props = section_properties_table(selection)
    st.subheader("Section properties")
    st.table(
        {
            "field": [k for k in props.keys() if k not in {"source"}],
            "value": [props[k] for k in props.keys() if k not in {"source"}],
        }
    )
    st.caption(f"Source: {props.get('source', 'n/a')}")
    st.caption("Notes: preliminary catalog values for engineering sandbox use.")
    if str(props.get("source", "")).strip().lower() in {"estimated", "assumed", "unknown"}:
        st.warning("review_required: verify section provenance before reporting.")
