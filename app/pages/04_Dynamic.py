from __future__ import annotations

import streamlit as st

from app.components.dynamic_viewer import (
    render_dynamic_charts,
    render_dynamic_settings,
    render_dynamic_summary,
    render_ground_motion_settings,
)
from app.services import dynamic_service, frame_service

st.title("Dynamic")
st.warning("Dynamic analysis shown here is preliminary for engineering exploration and not a normative seismic design verification.")

frame_input = st.session_state.get("frame_input")
if frame_input is None:
    st.error("Missing frame_input. Configure Frame page first.")
    st.stop()

gm_settings = render_ground_motion_settings()
dyn_settings = render_dynamic_settings()

if st.button("Run dynamic analysis", type="primary"):
    try:
        model = frame_service.build_frame_model(frame_input)
        ground_motion = dynamic_service.make_sine_ground_motion(**gm_settings)
        with st.spinner("Running dynamic integration..."):
            dynamic_result = dynamic_service.run_dynamic_case(model, frame_input, dyn_settings, ground_motion)
        st.session_state["dynamic_result"] = dynamic_result
    except RuntimeError as exc:
        st.error(f"Convergence error during dynamic analysis: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Dynamic run failed: {exc}")

dynamic_result = st.session_state.get("dynamic_result")
if dynamic_result is not None:
    render_dynamic_summary(dynamic_result)
    render_dynamic_charts(dynamic_result)
