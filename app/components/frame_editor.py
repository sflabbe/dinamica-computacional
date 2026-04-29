from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.services import FrameInput


def render_frame_input(key_prefix: str = "frame") -> FrameInput:
    section_selection = st.session_state.get("section_selection")
    width = st.number_input("Width L [m]", min_value=0.5, value=6.0, step=0.5, key=f"{key_prefix}_width")
    height = st.number_input("Height H [m]", min_value=0.5, value=3.0, step=0.5, key=f"{key_prefix}_height")
    n_col = st.number_input("Column discretization", min_value=1, value=4, step=1, key=f"{key_prefix}_n_col")
    n_beam = st.number_input("Beam discretization", min_value=1, value=6, step=1, key=f"{key_prefix}_n_beam")
    mass_total = st.number_input("Total mass [kg]", min_value=0.0, value=0.0, step=100.0, key=f"{key_prefix}_mass")
    damping_ratio = st.number_input(
        "Damping ratio [-]", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key=f"{key_prefix}_damp"
    )

    return FrameInput(
        width=float(width),
        height=float(height),
        n_col=int(n_col),
        n_beam=int(n_beam),
        section=section_selection,
        mass_total=float(mass_total),
        damping_ratio=float(damping_ratio),
    )


def render_frame_summary(frame_input: FrameInput) -> None:
    st.subheader("Frame input summary")
    st.json(
        {
            "width": frame_input.width,
            "height": frame_input.height,
            "n_col": frame_input.n_col,
            "n_beam": frame_input.n_beam,
            "mass_total": frame_input.mass_total,
            "damping_ratio": frame_input.damping_ratio,
            "section": None if frame_input.section is None else frame_input.section.__dict__,
        }
    )


def render_frame_preview(frame_input: FrameInput) -> None:
    x = [0.0, 0.0, frame_input.width, frame_input.width, 0.0]
    y = [0.0, frame_input.height, frame_input.height, 0.0, 0.0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="frame", line={"width": 3}))
    fig.update_layout(
        title="2D frame preview",
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        yaxis_scaleanchor="x",
        template="plotly_white",
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)
