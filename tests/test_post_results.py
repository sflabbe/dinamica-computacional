import numpy as np

from dc_solver.examples.portal_frame import build_portal_beam_hinge
from dc_solver.post.results import dynamic_result_from_dict, frame_state_from_model


def test_dynamic_result_missing_a_input_g_is_tolerated():
    raw = {
        "t": np.array([0.0, 0.1]),
        "ag": np.array([0.0, 0.2]),
        "drift": np.array([0.0, 0.01]),
        "Vb": np.array([0.0, 1.0]),
        "u": np.zeros((2, 6)),
    }
    out = dynamic_result_from_dict(raw)
    assert "warnings" in out.meta
    assert any("A_input_g missing" in w for w in out.meta["warnings"])


def test_hinge_hist_len_t_minus_1_aligns_to_t1():
    raw = {
        "t": np.array([0.0, 0.1, 0.2]),
        "ag": np.zeros(3),
        "drift": np.zeros(3),
        "Vb": np.zeros(3),
        "u": np.zeros((3, 2)),
        "hinge_hist": [
            [{"M": 1.0, "dtheta": 0.01, "a": 0.0, "name": "h0"}],
            [{"M": 2.0, "dtheta": 0.02, "a": 0.1, "name": "h0"}],
        ],
    }
    out = dynamic_result_from_dict(raw)
    assert len(out.hinges) == 1
    np.testing.assert_allclose(out.hinges[0].t, raw["t"][1:])


def test_frame_state_shapes():
    model, _ = build_portal_beam_hinge()
    u = np.zeros(model.ndof(), dtype=float)
    res = frame_state_from_model(model, u)
    assert res.node_xy_ref.shape == res.node_xy_def.shape
    assert res.node_xy_ref.shape[1] == 2
    assert res.node_umag.shape[0] == len(model.nodes)
    assert res.member_sigma_max.shape[0] == len(model.beams)
