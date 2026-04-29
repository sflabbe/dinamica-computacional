from __future__ import annotations

import streamlit as st

from app.components.frame_editor import render_frame_input, render_frame_preview, render_frame_summary
from app.services import frame_service

st.title("Frame")

section_selection = st.session_state.get("section_selection")
if section_selection is None:
    st.warning("No section selected yet. Go to Sections page first.")

frame_input = render_frame_input(key_prefix="frame_page")
frame_input.section = section_selection
st.session_state["frame_input"] = frame_input

render_frame_summary(frame_input)
render_frame_preview(frame_input)

try:
    model = frame_service.build_frame_model(frame_input)
    summary = frame_service.frame_summary(model)
    st.subheader("Model summary")
    st.table({"metric": list(summary.keys()), "value": list(summary.values())})
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build frame model: {exc}")
