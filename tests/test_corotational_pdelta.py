import numpy as np
from numpy.testing import assert_allclose

from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model
from dc_solver.fem.nodes import DofManager, Node
from dc_solver.static.newton import solve_static_newton

from tests.fixtures import assemble_full


def _build_column_model(L: float = 3.0,
                        E: float = 210e9,
                        A: float = 0.02,
                        I: float = 4e-4,
                        nlgeom: bool = False) -> Model:
    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, L, dm.new_trans(), dm.new_rot()),
    ]
    element = FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)
    fixed = np.array([*nodes[0].dof_u, nodes[0].dof_th], dtype=int)
    ndof = dm.ndof
    return Model(
        nodes=nodes,
        beams=[element],
        hinges=[],
        fixed_dofs=fixed,
        mass_diag=np.zeros(ndof),
        C_diag=np.zeros(ndof),
        load_const=np.zeros(ndof),
        col_hinge_groups=[],
        nlgeom=nlgeom,
    )


def test_pdelta_geometric_stiffness_matrix():
    L = 3.0
    E = 210e9
    A = 0.02
    I = 4e-4
    N = 1.2e6

    dm = DofManager()
    nodes = [
        Node(0.0, 0.0, dm.new_trans(), dm.new_rot()),
        Node(0.0, L, dm.new_trans(), dm.new_rot()),
    ]
    beam = FrameElementLinear2D(0, 1, E=E, A=A, I=I, nodes=nodes)
    k_geo = beam.k_geo_local(N)
    coeff = N / (30.0 * L)
    k_expected = coeff * np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 36.0, 3.0 * L, 0.0, -36.0, 3.0 * L],
        [0.0, 3.0 * L, 4.0 * L * L, 0.0, -3.0 * L, -1.0 * L * L],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -36.0, -3.0 * L, 0.0, 36.0, -3.0 * L],
        [0.0, 3.0 * L, -1.0 * L * L, 0.0, -3.0 * L, 4.0 * L * L],
    ])
    assert_allclose(k_geo, k_expected, rtol=1e-12, atol=1e-12)


def test_pdelta_increases_lateral_deflection():
    model_lin = _build_column_model(nlgeom=False)
    model_geo = _build_column_model(nlgeom=True)

    top = model_lin.nodes[1]
    ux = top.dof_u[0]
    uy = top.dof_u[1]

    P = -5_000_000.0
    H = 5_000.0
    model_lin.load_const[ux] = H
    model_lin.load_const[uy] = P
    model_geo.load_const[:] = model_lin.load_const

    u_lin = solve_static_newton(model_lin, model_lin.load_const)
    u_geo = solve_static_newton(model_geo, model_geo.load_const)

    K_lin_red, _, _ = model_lin.assemble(u_lin, u_lin)
    K_geo_red, _, _ = model_geo.assemble(u_geo, u_geo)
    assert K_geo_red[0, 0] < K_lin_red[0, 0]

    u_eff = np.zeros_like(u_lin)
    u_eff[model_geo.free_dofs()] = np.linalg.solve(
        K_geo_red + 1e-14 * np.eye(K_geo_red.shape[0]),
        model_geo.load_const[model_geo.free_dofs()],
    )
    assert abs(u_eff[ux]) > abs(u_lin[ux])

    K_geo, _ = assemble_full(model_geo, u_geo)
    assert_allclose(K_geo, K_geo.T, rtol=1e-6, atol=1e-8)
