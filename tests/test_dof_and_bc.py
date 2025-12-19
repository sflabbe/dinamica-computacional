import numpy as np
from numpy.testing import assert_allclose, assert_equal

from dinamica_computacional.core.dof import DofManager, Node
from dinamica_computacional.core.model import Model, ModelOptions
from dinamica_computacional.elements.beam2d import Beam2D

from tests.fixtures import assemble_full


def test_dof_mapping_and_boundary_conditions():
    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(2.0, 0.0, dm.new_trans(), dm.new_rot()),
    ]
    element = Beam2D(0, 1, E=210e9, A=0.01, I=1e-4, nodes=nodes)
    fixed = np.array([*nodes[0].dof_u, nodes[0].dof_th], dtype=int)
    ndof = dm.ndof
    model = Model(
        nodes=nodes,
        elements=[element],
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=np.ones(ndof),
        C_diag=np.zeros(ndof),
        load_const=np.zeros(ndof),
        options=ModelOptions(),
    )

    assert_equal(model.ndof(), 6)
    free = model.free_dofs()
    assert_equal(free.size, 3)
    assert_equal(free, np.array([nodes[1].dof_u[0], nodes[1].dof_u[1], nodes[1].dof_th]))

    K_full, _ = assemble_full(model, np.zeros(ndof))
    K_red, _, _ = model.assemble(np.zeros(ndof), np.zeros(ndof))
    assert_allclose(K_red, K_full[np.ix_(free, free)])

    load_free = np.array([1.0, -2.0, 0.5])
    u_free = np.linalg.solve(K_red + 1e-14 * np.eye(free.size), load_free)
    assert_allclose(K_red @ u_free, load_free)

    assert_allclose(model.mass_diag[free], np.ones(3))
