import numpy as np
from scipy.linalg import eigh

from dc_solver.modal.eigensolver import solve_eigenpairs


def test_1dof_dense_backend_frequency():
    K = np.array([[10.0]])
    M = np.array([[2.0]])
    omega, phi = solve_eigenpairs(K, M, n_modes=1)
    assert np.isclose(omega[0], np.sqrt(5.0))
    assert np.isclose(phi.T @ M @ phi, 1.0)


def test_2dof_matches_scipy_eigh():
    K = np.array([[6.0, -2.0], [-2.0, 4.0]])
    M = np.array([[2.0, 0.0], [0.0, 1.0]])
    vals, _ = eigh(K, M)
    w_ref = np.sqrt(np.clip(vals, 0.0, None))
    w, _ = solve_eigenpairs(K, M, n_modes=2)
    assert np.allclose(w, w_ref)
