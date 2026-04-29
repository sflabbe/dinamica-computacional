from __future__ import annotations

import streamlit as st

from app.services.dynamic_service import dynamic_summary
from app.services.session_schema import AnalysisSettings
from dc_solver.post.plotly_charts import plot_base_shear_drift, plot_drift_time_history, plot_energy_balance


def render_ground_motion_settings(key_prefix: str = "gm") -> dict:
    col1, col2 = st.columns(2)
    with col1:
        amplitude_g = st.number_input("Amplitude [g]", min_value=0.001, value=0.15, key=f"{key_prefix}_amp")
        freq_hz = st.number_input("Frequency [Hz]", min_value=0.1, value=1.5, key=f"{key_prefix}_freq")
    with col2:
        duration = st.number_input("Duration [s]", min_value=0.5, value=10.0, key=f"{key_prefix}_dur")
        dt = st.number_input("dt [s]", min_value=0.001, value=0.01, key=f"{key_prefix}_dt")
    return {"amplitude_g": float(amplitude_g), "freq_hz": float(freq_hz), "duration": float(duration), "dt": float(dt)}


def render_dynamic_settings(key_prefix: str = "dyn") -> AnalysisSettings:
    integrator = st.selectbox("Integrator", options=["hht", "newmark"], index=0, key=f"{key_prefix}_int")
    return AnalysisSettings(run_dynamic=True, run_modal=False, integrator=str(integrator))


def render_dynamic_summary(dynamic_result) -> None:
    summary = dynamic_summary(dynamic_result)
    st.subheader("Dynamic summary")
    st.dataframe([summary], use_container_width=True)


def render_dynamic_charts(dynamic_result) -> None:
    st.plotly_chart(plot_drift_time_history(dynamic_result), use_container_width=True)
    st.plotly_chart(plot_base_shear_drift(dynamic_result), use_container_width=True)
    if dynamic_result.energy:
        st.plotly_chart(plot_energy_balance(dynamic_result), use_container_width=True)
