import numpy as np
from numpy.testing import assert_allclose

from dc_solver.fem.nodes import DofManager, Node
from dc_solver.hinges.models import BilinearMThetaHinge1D, RotSpringElement, SHMBeamHinge1D


def _nodes():
    dm = DofManager()
    return [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
    ]


def _build_bilinear_hinge(K0: float = 10_000.0, My: float = 100.0) -> RotSpringElement:
    hinge = BilinearMThetaHinge1D(K0=K0, My=My, alpha_post=0.0)
    return RotSpringElement(0, 1, "beam_bilinear", None, hinge, _nodes())


def _build_shm_hinge(K0: float = 10_000.0, My: float = 100.0) -> RotSpringElement:
    hinge = SHMBeamHinge1D(K0_0=K0, My_0=My, alpha_post=0.0, cK=0.0, cMy=0.0)
    return RotSpringElement(0, 1, "beam_shm", None, hinge, _nodes())


def test_mtheta_elastic_plastic_response_and_commit():
    hinge_el = _build_bilinear_hinge()
    nd = 6
    u_comm = np.zeros(nd)

    # Elastic trial must not mutate committed state.
    u_trial = u_comm.copy()
    u_trial[5] = 0.005
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(info["M"], 10_000.0 * 0.005)
    assert hinge_el.beam_hinge.M_comm == 0.0

    hinge_el.commit()
    assert_allclose(hinge_el.beam_hinge.M_comm, 10_000.0 * 0.005)

    # Perfect-plastic return mapping caps the moment at My.
    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] = 0.02
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(abs(info["M"]), hinge_el.beam_hinge.My, rtol=1e-12)

    # Unloading is elastic with slope K0 from the committed plastic state.
    hinge_el.commit()
    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] -= 0.002
    M_prev = hinge_el.beam_hinge.M_comm
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(info["M"], M_prev - hinge_el.beam_hinge.K0 * 0.002)


def test_shm_contract_is_smooth_bouc_wen_not_exact_return_mapping():
    hinge_el = _build_shm_hinge()
    nd = 6
    u_comm = np.zeros(nd)

    # Small rotations should recover the initial tangent approximately.
    u_trial = u_comm.copy()
    u_trial[5] = 1e-5
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(info["M"], 10_000.0 * 1e-5, rtol=1e-4, atol=1e-8)
    assert hinge_el.beam_hinge.M_comm == 0.0

    hinge_el.commit()
    assert_allclose(hinge_el.beam_hinge.M_comm, info["M"], rtol=1e-12)

    # At larger rotations, SHM remains bounded by its internal z clamp, but it is
    # not expected to sit exactly on My like a perfect-plastic return mapper.
    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] = 0.02
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert np.isfinite(info["M"])
    assert abs(info["M"]) <= 1.2 * hinge_el.beam_hinge.My_0 + 1e-9
    assert not np.isclose(abs(info["M"]), hinge_el.beam_hinge.My_0, rtol=1e-6, atol=1e-9)
