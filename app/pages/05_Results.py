from __future__ import annotations

import json

import streamlit as st

from app.components.result_cards import metric_card, render_analysis_warnings
from dc_solver.reporting.run_info import to_jsonable

st.title("Results")

section_selection = st.session_state.get("section_selection")
frame_input = st.session_state.get("frame_input")
modal_result = st.session_state.get("modal_result")
dynamic_result = st.session_state.get("dynamic_result")

col1, col2, col3 = st.columns(3)
with col1:
    metric_card("Section selected", "yes" if section_selection is not None else "no")
with col2:
    metric_card("Frame configured", "yes" if frame_input is not None else "no")
with col3:
    metric_card("Analyses", f"modal: {'yes' if modal_result is not None else 'no'} / dynamic: {'yes' if dynamic_result is not None else 'no'}")

warnings: list[str] = []
if modal_result is not None:
    warnings.extend([str(w) for w in modal_result.meta.get("warnings", [])])
if dynamic_result is not None:
    warnings.extend([str(w) for w in dynamic_result.meta.get("warnings", [])])
render_analysis_warnings(warnings)

payload = {
    "section_selection": section_selection,
    "frame_input": frame_input,
    "modal_result": modal_result,
    "dynamic_result": dynamic_result,
}
json_payload = json.dumps(to_jsonable(payload), indent=2)
st.download_button("Download lightweight JSON", data=json_payload, file_name="dc_results.json", mime="application/json")
