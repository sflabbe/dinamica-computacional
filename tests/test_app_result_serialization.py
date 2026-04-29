from __future__ import annotations

import json

from app.services.dynamic_service import make_sine_ground_motion
from app.services.frame_service import build_frame_model
from app.services.modal_service import run_modal_case
from app.services.session_schema import AnalysisSettings, FrameInput
from dc_solver.post.results import dynamic_result_from_dict
from dc_solver.reporting.run_info import to_jsonable


def test_lightweight_result_json_serialization() -> None:
    frame_input = FrameInput(width=5.0, height=3.0)
    model = build_frame_model(frame_input)
    modal_result = run_modal_case(model, AnalysisSettings(n_modes=2))
    dynamic_result = dynamic_result_from_dict({"t": [0.0, 0.1], "ag": [0.0, 0.01], "drift": [0.0, 0.0], "Vb": [0.0, 0.0]})

    payload = {
        "frame_input": frame_input,
        "modal_result": modal_result,
        "dynamic_result": dynamic_result,
    }
    text = json.dumps(to_jsonable(payload))
    assert isinstance(text, str)
    assert "modal_result" in text


def test_ground_motion_builder_still_works() -> None:
    gm = make_sine_ground_motion(amplitude_g=0.2, freq_hz=2.0, duration=1.0, dt=0.01)
    assert len(gm["t"]) == len(gm["ag"])
    assert gm["t"][0] == 0.0
