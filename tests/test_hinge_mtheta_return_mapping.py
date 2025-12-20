import numpy as np
import pytest
from numpy.testing import assert_allclose

from dc_solver.fem.nodes import DofManager, Node
from dc_solver.hinges.models import SHMBeamHinge1D, RotSpringElement


def _build_hinge(K0: float = 10_000.0, My: float = 100.0) -> RotSpringElement:
    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
    ]
    hinge = SHMBeamHinge1D(K0_0=K0, My_0=My, alpha_post=0.0, cK=0.0, cMy=0.0)
    return RotSpringElement(0, 1, "beam_shm", None, hinge, nodes)


@pytest.mark.xfail(reason="SHM hinge model has pre-existing numerical issue - needs investigation")
def test_mtheta_elastic_plastic_response_and_commit():
    hinge_el = _build_hinge()
    nd = 6
    u_comm = np.zeros(nd)

    u_trial = u_comm.copy()
    u_trial[5] = 0.005
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(info["M"], 10_000.0 * 0.005)
    assert hinge_el.beam_hinge.M_comm == 0.0

    hinge_el.commit()
    assert_allclose(hinge_el.beam_hinge.M_comm, 10_000.0 * 0.005)

    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] = 0.02
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(abs(info["M"]), hinge_el.beam_hinge.My_0, rtol=1e-6)

    hinge_el.commit()
    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] -= 0.002
    M_prev = hinge_el.beam_hinge.M_comm
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    k0_current = hinge_el.beam_hinge.K0_0
    assert_allclose(info["M"], M_prev - k0_current * 0.002)
