import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.integrators.static_newton import solve_static_newton

from tests.fixtures import assemble_full, build_cantilever_model, build_portal_frame_model


def test_linear_cantilever_tip_load():
    L = 3.0
    E = 210e9
    I = 6e-4
    P = -10_000.0

    model = build_cantilever_model(L=L, E=E, A=0.02, I=I)
    tip_ux = model.nodes[1].dof_u[0]
    model.load_const[tip_ux] = P

    u = solve_static_newton(model)
    delta = P * L**3 / (3.0 * E * I)

    assert_allclose(u[tip_ux], delta, rtol=1e-3)

    K_full, _ = assemble_full(model, np.zeros(model.ndof()))
    free = model.free_dofs()
    K_red = K_full[np.ix_(free, free)]
    u_lin = np.zeros(model.ndof())
    u_lin[free] = np.linalg.solve(K_red + 1e-14 * np.eye(free.size), model.load_const[free])
    assert_allclose(u, u_lin, rtol=1e-6, atol=1e-12)


def test_portal_gravity_reactions_balance():
    model = build_portal_frame_model(L=4.0, H=3.0)
    uy2 = model.nodes[2].dof_u[1]
    uy3 = model.nodes[3].dof_u[1]
    model.load_const[uy2] = -50_000.0
    model.load_const[uy3] = -50_000.0

    u = solve_static_newton(model)

    max_uy = np.max(np.abs(u[[uy2, uy3]]))
    assert max_uy > 0.0

    K_full, Rint = assemble_full(model, u)
    reactions = Rint - model.load_const

    fixed = model.fixed_dofs
    fixed_fy = [d for d in fixed if d in (model.nodes[0].dof_u[1], model.nodes[1].dof_u[1])]
    sum_reaction_fy = np.sum(reactions[fixed_fy])
    sum_loads_fy = np.sum(model.load_const[[uy2, uy3]])
    assert_allclose(sum_reaction_fy + sum_loads_fy, 0.0, atol=1e-6)
    assert_allclose(K_full, K_full.T)
