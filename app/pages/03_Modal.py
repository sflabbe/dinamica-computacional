from __future__ import annotations

import streamlit as st

from app.components.modal_viewer import (
    render_modal_settings,
    render_modal_summary,
    render_mode_shape_chart,
    render_spectrum_settings,
)
from app.services import frame_service, modal_service

st.title("Modal")
st.warning("Modal analysis is preliminary and uses the current assembled tangent stiffness. It is not an EC8 design check; manual verification is required for code compliance.")

frame_input = st.session_state.get("frame_input")
if frame_input is None:
    st.error("Missing frame_input. Configure Frame page first.")
    st.stop()

settings = render_modal_settings()
render_spectrum_settings()

if st.button("Run modal analysis", type="primary"):
    try:
        model = frame_service.build_frame_model(frame_input)
        modal_result = modal_service.run_modal_case(model, settings)
        st.session_state["modal_result"] = modal_result
    except Exception as exc:  # noqa: BLE001
        st.error(f"Modal run failed: {exc}")

modal_result = st.session_state.get("modal_result")
if modal_result is not None:
    render_modal_summary(modal_result)
    mode_index = st.slider("Mode index", min_value=1, max_value=max(1, len(modal_result.freq_hz)), value=1)
    render_mode_shape_chart(modal_result, mode_index=int(mode_index - 1))
