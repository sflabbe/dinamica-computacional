from app.services.frame_service import build_frame_model
from app.services.modal_service import run_modal_case
from app.services.session_schema import AnalysisSettings, FrameInput


def test_modal_case_has_positive_frequencies():
    model = build_frame_model(FrameInput(width=5.0, height=3.0))
    res = run_modal_case(model, AnalysisSettings(n_modes=3))
    assert len(res.freq_hz) >= 1
    assert float(res.freq_hz[0]) > 0.0
