from __future__ import annotations

import numpy as np

from dc_solver.integrators import solve_dynamic
from dc_solver.post.results import dynamic_result_from_dict

from .session_schema import AnalysisSettings, FrameInput


def make_sine_ground_motion(amplitude_g: float, freq_hz: float, duration: float, dt: float):
    t = np.arange(0.0, float(duration) + 0.5 * float(dt), float(dt), dtype=float)
    ag = float(amplitude_g) * np.sin(2.0 * np.pi * float(freq_hz) * t)
    return {"t": t, "ag": ag}


def run_dynamic_case(model, frame_input: FrameInput, settings: AnalysisSettings, ground_motion):
    t = np.asarray(ground_motion["t"], dtype=float)
    ag = np.asarray(ground_motion["ag"], dtype=float)
    raw = solve_dynamic(
        settings.integrator,
        model=model,
        t=t,
        ag=ag,
        drift_height=float(frame_input.height),
        base_nodes=(0, 1),
        drift_nodes=(2, 3),
    )
    if "t" not in raw:
        raw["t"] = t
    if "ag" not in raw:
        raw["ag"] = ag
    return dynamic_result_from_dict(raw, name=f"dynamic_{settings.integrator}")


def dynamic_summary(result) -> dict[str, float]:
    n_steps = float(len(result.t))
    max_drift = float(np.max(np.abs(result.drift))) if len(result.drift) else 0.0
    max_vb = float(np.max(np.abs(result.Vb))) if len(result.Vb) else 0.0
    max_ag = float(np.max(np.abs(result.ag))) if len(result.ag) else 0.0
    return {"n_steps": n_steps, "max_abs_drift": max_drift, "max_abs_Vb": max_vb, "max_abs_ag": max_ag}
