from app.services.frame_service import build_frame_model, frame_summary
from app.services.session_schema import FrameInput
from dc_solver.fem.model import Model


def test_build_frame_model_returns_model():
    model = build_frame_model(FrameInput(width=5.0, height=3.0))
    assert isinstance(model, Model)


def test_frame_summary_has_expected_keys():
    model = build_frame_model(FrameInput(width=5.0, height=3.0))
    summary = frame_summary(model)
    assert summary["n_nodes"] > 0
    assert summary["n_beams"] > 0
    assert summary["ndof"] > 0
