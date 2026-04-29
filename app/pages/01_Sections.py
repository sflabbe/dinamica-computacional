from __future__ import annotations

import streamlit as st

from app.components.section_viewer import render_section_properties, render_section_selector

st.title("Sections")
selection = render_section_selector(key_prefix="sections_page")
st.session_state["section_selection"] = selection
render_section_properties(selection)

st.info("Disclaimer: values are for engineering sandbox usage only.")
