import numpy as np
from numpy.testing import assert_allclose

from dc_solver.post.plotting import beam_local_displacements


def test_beam_shape_functions_match_boundary_conditions():
    L = 2.0
    u_local = np.array([0.05, -0.02, 0.01, 0.02, 0.03, -0.04])
    xi = np.array([0.0, 1.0])
    _, v, dv_dx = beam_local_displacements(xi, L, u_local)

    assert_allclose(v[0], u_local[1], rtol=0.0, atol=1e-12)
    assert_allclose(v[1], u_local[4], rtol=0.0, atol=1e-12)
    assert_allclose(dv_dx[0], u_local[2], rtol=0.0, atol=1e-12)
    assert_allclose(dv_dx[1], u_local[5], rtol=0.0, atol=1e-12)
