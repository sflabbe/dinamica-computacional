import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.core.dof import DofManager, Node
from dinamica_computacional.elements.hinge_mtheta import SHMBeamHinge1D, RotSpringElementMTheta


def _build_hinge(K0: float = 10_000.0, My: float = 100.0) -> RotSpringElementMTheta:
    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
    ]
    hinge = SHMBeamHinge1D(K0_0=K0, My_0=My, alpha_post=0.0)
    return RotSpringElementMTheta(0, 1, hinge, nodes)


def test_mtheta_elastic_plastic_response_and_commit():
    hinge_el = _build_hinge()
    nd = 6
    u_comm = np.zeros(nd)

    u_trial = u_comm.copy()
    u_trial[5] = 0.005
    k_l, f_l, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(info["M"], 10_000.0 * 0.005)
    assert hinge_el.hinge.M_comm == 0.0

    hinge_el.commit()
    assert_allclose(hinge_el.hinge.M_comm, info["M_new"])

    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] = 0.02
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(abs(info["M"]), hinge_el.hinge.My_0, rtol=1e-6)

    hinge_el.commit()
    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] -= 0.002
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    k0_current = hinge_el.hinge.K0_0 * np.exp(-hinge_el.hinge.cK * hinge_el.hinge.a_comm)
    assert_allclose(info["M"], hinge_el.hinge.M_comm - k0_current * 0.002)
