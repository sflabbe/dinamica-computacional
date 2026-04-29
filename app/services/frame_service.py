from __future__ import annotations

import numpy as np

from dc_solver.examples.portal_frame import build_portal_beam_hinge
from dc_solver.utils.gravity import solve_gravity_only

from .session_schema import FrameInput


def build_frame_model(frame_input: FrameInput):
    model, _ctx = build_portal_beam_hinge(
        H=float(frame_input.height),
        L=float(frame_input.width),
        nseg=int(frame_input.n_beam),
        mass_mode="distributed",
    )
    if frame_input.mass_total > 0.0:
        m = np.asarray(model.mass_diag, dtype=float).copy()
        msum = float(np.sum(m))
        if msum > 0:
            model.mass_diag = m * (float(frame_input.mass_total) / msum)
    return model


def run_gravity_case(model):
    return solve_gravity_only(model)


def frame_summary(model) -> dict[str, int | float]:
    return {
        "n_nodes": int(len(model.nodes)),
        "n_beams": int(len(model.beams)),
        "n_hinges": int(len(model.hinges)),
        "ndof": int(model.ndof()),
    }
