import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.elements.beam2d import Beam2D
from dinamica_computacional.core.dof import DofManager, Node


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
    beam = Beam2D(0, 1, E=E, A=A, I=I, nodes=nodes)

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
