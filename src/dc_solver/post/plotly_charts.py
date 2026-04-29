from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from dc_solver.modal import ModalResults, SpectrumEC8
from .results import DynamicResult, FrameStateResult, HingeTimeHistory


def plot_frame_state(state: FrameStateResult):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=state.node_xy_ref[:, 0], y=state.node_xy_ref[:, 1], mode="markers", name="ref"))
    fig.add_trace(go.Scatter(x=state.node_xy_def[:, 0], y=state.node_xy_def[:, 1], mode="markers", name="def"))
    fig.update_layout(title=f"Frame state: {state.label}", xaxis_title="x", yaxis_title="y")
    return fig


def plot_drift_time_history(result: DynamicResult):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.t, y=result.drift, mode="lines", name="drift"))
    fig.update_layout(title="Drift vs time", xaxis_title="t", yaxis_title="drift")
    return fig


def plot_base_shear_drift(result: DynamicResult):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.drift, y=result.Vb, mode="lines", name="Vb-drift"))
    fig.update_layout(title="Base shear vs drift", xaxis_title="drift", yaxis_title="Vb")
    return fig


def plot_hinge_mtheta(history: HingeTimeHistory):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.dtheta, y=history.M, mode="lines", name=history.name))
    fig.update_layout(title=f"Hinge M-θ: {history.name}", xaxis_title="dtheta", yaxis_title="M")
    return fig


def plot_energy_balance(result: DynamicResult):
    fig = go.Figure()
    for k, v in result.energy.items():
        n = min(len(result.t), len(v))
        fig.add_trace(go.Scatter(x=result.t[:n], y=np.asarray(v)[:n], mode="lines", name=str(k)))
    fig.update_layout(title="Energy balance", xaxis_title="t", yaxis_title="energy")
    return fig


def plot_mode_shape(modal: ModalResults, mode_index=0, *, scale=1.0):
    vec = modal.modes_full[:, mode_index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(vec.size), y=vec * scale, mode="lines+markers", name=f"mode {mode_index+1}"))
    fig.update_layout(title=f"Mode shape {mode_index+1}", xaxis_title="DOF", yaxis_title="Amplitude")
    return fig


def plot_response_spectrum(spectrum: SpectrumEC8, *, t_min=0.0, t_max=4.0, n=400):
    T = np.linspace(t_min, t_max, n)
    Sa = spectrum.Sa(T)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=Sa, mode="lines", name="Sa(T)"))
    fig.update_layout(title="Elastic response spectrum (helper)", xaxis_title="T [s]", yaxis_title="Sa [m/s²]")
    return fig
