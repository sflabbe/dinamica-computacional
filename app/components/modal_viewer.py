from __future__ import annotations

import streamlit as st

from app.services.modal_service import modal_summary_table
from app.services.session_schema import AnalysisSettings
from dc_solver.post.plotly_charts import plot_mode_shape, plot_response_spectrum
from dc_solver.modal import SpectrumEC8


def render_modal_settings(key_prefix: str = "modal") -> AnalysisSettings:
    n_modes = st.number_input("Number of modes", min_value=1, max_value=24, value=6, step=1, key=f"{key_prefix}_n_modes")
    run_gravity = st.checkbox("Run gravity preload", value=True, key=f"{key_prefix}_gravity")
    return AnalysisSettings(run_gravity=run_gravity, run_modal=True, run_dynamic=False, n_modes=int(n_modes))


def render_modal_summary(modal_result) -> None:
    rows = modal_summary_table(modal_result)
    st.subheader("Modal summary")
    st.dataframe(rows, use_container_width=True)


def render_mode_shape_chart(modal_result, mode_index: int = 0) -> None:
    fig = plot_mode_shape(modal_result, mode_index=mode_index)
    st.plotly_chart(fig, use_container_width=True)


def render_spectrum_settings(key_prefix: str = "spectrum") -> dict:
    col1, col2 = st.columns(2)
    with col1:
        ag = st.number_input("ag [m/s²]", min_value=0.1, value=2.5, key=f"{key_prefix}_ag")
        S = st.number_input("Soil factor S", min_value=0.5, value=1.0, key=f"{key_prefix}_S")
    with col2:
        Tb = st.number_input("Tb [s]", min_value=0.01, value=0.15, key=f"{key_prefix}_Tb")
        Tc = st.number_input("Tc [s]", min_value=0.02, value=0.5, key=f"{key_prefix}_Tc")
        Td = st.number_input("Td [s]", min_value=0.1, value=2.0, key=f"{key_prefix}_Td")
    params = {"ag": float(ag), "S": float(S), "Tb": float(Tb), "Tc": float(Tc), "Td": float(Td)}
    if st.checkbox("Preview elastic spectrum", value=False, key=f"{key_prefix}_preview"):
        spectrum = SpectrumEC8(**params)
        st.plotly_chart(plot_response_spectrum(spectrum), use_container_width=True)
    return params
