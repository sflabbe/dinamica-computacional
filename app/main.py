from __future__ import annotations

import streamlit as st

from app import services

st.set_page_config(page_title="dc-solver", page_icon="📐", layout="centered")

st.title("dc-solver")
st.caption("engineering sandbox, not a prüffähiger Nachweis")

st.markdown(
    """
    ### Fase 4B UI
    1. Use **Sections** to select a section profile.
    2. Use **Frame** to define a 2D frame and preview it.
    3. Modal and dynamic analyses are intentionally disabled in this phase.
    """
)

with st.expander("Smoke import: app.services", expanded=False):
    st.code(
        "\n".join(
            [
                "SectionSelection",
                "FrameInput",
                "available_section_families",
                "available_profiles",
                "section_properties_table",
                "build_frame_model",
                "frame_summary",
            ]
        )
    )
    st.write("Services module loaded:", bool(services.__all__))
