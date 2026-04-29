import numpy as np

from app.services.dynamic_service import dynamic_summary, make_sine_ground_motion
from dc_solver.post.results import dynamic_result_from_dict


def test_make_sine_ground_motion_has_matching_lengths():
    gm = make_sine_ground_motion(amplitude_g=0.2, freq_hz=1.5, duration=2.0, dt=0.01)
    assert len(gm["t"]) == len(gm["ag"])


def test_dynamic_summary_tolerates_minimal_result():
    result = dynamic_result_from_dict({"t": np.array([0.0]), "ag": np.array([0.0])})
    summary = dynamic_summary(result)
    assert summary["n_steps"] == 1.0
    assert summary["max_abs_drift"] == 0.0
