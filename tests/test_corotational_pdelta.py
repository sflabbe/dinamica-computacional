import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.core.dof import DofManager, Node
from dinamica_computacional.core.model import Model, ModelOptions
from dinamica_computacional.elements.beam2d import Beam2D
from dinamica_computacional.integrators.static_newton import solve_static_newton

from tests.fixtures import assemble_full


def _build_column_model(L: float = 3.0,
                        E: float = 210e9,
                        A: float = 0.02,
                        I: float = 4e-4) -> Model:
    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, L, dm.new_trans(), dm.new_rot()),
    ]
    element = Beam2D(0, 1, E=E, A=A, I=I, nodes=nodes)
    fixed = np.array([*nodes[0].dof_u, nodes[0].dof_th], dtype=int)
    ndof = dm.ndof
    return Model(
        nodes=nodes,
        elements=[element],
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=np.zeros(ndof),
        C_diag=np.zeros(ndof),
        load_const=np.zeros(ndof),
        options=ModelOptions(),
    )


def test_pdelta_increases_lateral_deflection():
    model_lin = _build_column_model()
    model_geo = _build_column_model()

    top = model_lin.nodes[1]
    ux = top.dof_u[0]
    uy = top.dof_u[1]

    P = -5_000_000.0
    H = 5_000.0
    model_lin.load_const[ux] = H
    model_lin.load_const[uy] = P
    model_geo.load_const[:] = model_lin.load_const

    for e in model_lin.elements:
        e.geometry = "linear"
    for e in model_geo.elements:
        e.geometry = "corotational"

    u_lin = solve_static_newton(model_lin)
    u_geo = solve_static_newton(model_geo)

    assert abs(u_geo[ux]) > abs(u_lin[ux]) * 1.05

    K_geo, _ = assemble_full(model_geo, u_geo)
    assert_allclose(K_geo, K_geo.T, rtol=1e-6, atol=1e-8)
