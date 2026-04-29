import numpy as np
from numpy.testing import assert_allclose

from dc_solver.post.plotting import beam_local_displacements
from dc_solver.post.plotting import _scaled_displacements
from dc_solver.fem.nodes import Node, DofManager
from dc_solver.fem.model import Model


def test_beam_shape_functions_match_boundary_conditions():
    L = 2.0
    u_local = np.array([0.05, -0.02, 0.01, 0.02, 0.03, -0.04])
    xi = np.array([0.0, 1.0])
    _, v, dv_dx = beam_local_displacements(xi, L, u_local)

    assert_allclose(v[0], u_local[1], rtol=0.0, atol=1e-12)
    assert_allclose(v[1], u_local[4], rtol=0.0, atol=1e-12)
    assert_allclose(dv_dx[0], u_local[2], rtol=0.0, atol=1e-12)
    assert_allclose(dv_dx[1], u_local[5], rtol=0.0, atol=1e-12)


def test_scaled_displacements_scales_shared_translations_once():
    dm = DofManager()
    n0 = Node(0.0, 0.0, dm.new_trans(), dm.new_rot())
    # Aux node sharing translations with n0, but with independent rotation DOF.
    n0_aux = Node(0.0, 0.0, n0.dof_u, dm.new_rot())
    model = Model(
        nodes=[n0, n0_aux],
        beams=[],
        hinges=[],
        fixed_dofs=np.array([], dtype=int),
        mass_diag=np.zeros(dm.ndof),
        C_diag=np.zeros(dm.ndof),
        load_const=np.zeros(dm.ndof),
    )

    u = np.array([1.5, -2.0, 7.0, -4.0], dtype=float)
    u_scaled = _scaled_displacements(model, u, scale=3.0)

    assert_allclose(u_scaled, np.array([4.5, -6.0, 7.0, -4.0]), rtol=0.0, atol=1e-12)
