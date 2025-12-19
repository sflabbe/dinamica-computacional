import numpy as np
from numpy.testing import assert_allclose

from dc_solver.fem.frame2d import FrameElementLinear2D, rot2d
from dc_solver.fem.nodes import DofManager, Node


def test_beam2d_linear_stiffness_matrix():
    L = 2.0
    E = 210e9
    A = 0.02
    I = 5e-4

    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(L, 0.0, dm.new_trans(), dm.new_rot()),
    ]
    beam = FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)

    dofs, K, _, _ = beam.stiffness_and_force_global(np.zeros(dm.ndof))
    assert dofs.size == 6

    k_ax = E * A / L
    k11 = 12 * E * I / (L ** 3)
    k12 = 6 * E * I / (L ** 2)
    k22 = 4 * E * I / L
    k22b = 2 * E * I / L

    assert_allclose(K[0, 0], k_ax)
    assert_allclose(K[0, 3], -k_ax)
    assert_allclose(K[3, 0], -k_ax)
    assert_allclose(K[3, 3], k_ax)

    assert_allclose(K[1, 1], k11)
    assert_allclose(K[1, 2], k12)
    assert_allclose(K[2, 2], k22)
    assert_allclose(K[1, 4], -k11)
    assert_allclose(K[2, 4], -k12)
    assert_allclose(K[5, 5], k22)
    assert_allclose(K[2, 5], k22b)

    assert_allclose(K, K.T)


def test_beam2d_global_transformation_matches_rotation():
    L = 2.0
    E = 210e9
    A = 0.02
    I = 5e-4

    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(L / np.sqrt(2.0), L / np.sqrt(2.0), dm.new_trans(), dm.new_rot()),
    ]
    beam = FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)
    Lc, c, s = beam._geom()
    k_local = beam.k_local()
    T = rot2d(c, s)
    k_expected = T.T @ k_local @ T

    dofs, K, _, _ = beam.stiffness_and_force_global(np.zeros(dm.ndof))
    assert dofs.size == 6
    assert_allclose(K, k_expected, rtol=1e-12, atol=1e-9)
