from __future__ import annotations

import streamlit as st


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def render_analysis_warnings(warnings: list[str]) -> None:
    if not warnings:
        st.success("No warnings reported by analyses.")
        return
    for warning in warnings:
        st.warning(warning)
