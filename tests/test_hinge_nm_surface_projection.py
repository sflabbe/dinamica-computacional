import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.core.dof import DofManager, Node
from dinamica_computacional.elements.hinge_nm import ColumnHingeNMRot, RotSpringElementNM
from dinamica_computacional.materials.nm_surface import NMSurfacePolygon, moment_capacity_from_polygon


def _build_nm_hinge() -> RotSpringElementNM:
    surface = NMSurfacePolygon(vertices=[
        (-200.0, -150.0),
        (-200.0, 150.0),
        (0.0, 200.0),
        (200.0, 150.0),
        (200.0, -150.0),
        (0.0, -200.0),
    ])
    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
    ]
    hinge = ColumnHingeNMRot(surface=surface, k0=10_000.0, alpha_post=0.0)
    return RotSpringElementNM(0, 1, hinge, nodes)


def test_nm_surface_projection_caps_moment():
    hinge_el = _build_nm_hinge()
    nd = 6
    u_comm = np.zeros(nd)

    hinge_el.set_yield_from_N(0.0)
    My = hinge_el.hinge.My_comm
    u_trial = u_comm.copy()
    u_trial[5] = 0.05
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(abs(info["M"]), My, rtol=1e-6)
    assert hinge_el.hinge.M_comm == 0.0

    hinge_el.commit()
    u_comm = u_trial.copy()
    u_trial = u_comm.copy()
    u_trial[5] += 0.05
    _, _, info = hinge_el.eval_trial(u_trial, u_comm)
    assert_allclose(abs(info["M"]), My, rtol=1e-6)


def test_nm_surface_capacity_symmetry():
    surface = NMSurfacePolygon(vertices=[
        (-100.0, -50.0),
        (-100.0, 50.0),
        (0.0, 80.0),
        (100.0, 50.0),
        (100.0, -50.0),
        (0.0, -80.0),
    ])
    m_pos = moment_capacity_from_polygon(surface, 50.0)
    m_neg = moment_capacity_from_polygon(surface, -50.0)
    assert_allclose(m_pos, m_neg)
