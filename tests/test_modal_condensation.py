import numpy as np

from dc_solver.modal.eigensolver import condense_massless_dofs


def test_condense_and_expand_mode():
    K = np.array([[10.0, -4.0], [-4.0, 8.0]])
    Mdiag = np.array([2.0, 0.0])
    c = condense_massless_dofs(K, Mdiag)
    assert c.K_eff.shape == (1, 1)
    phi_a = np.array([[1.0]])
    phi = c.T_expand @ phi_a
    # static constraint for massless dof: Kba*phi_a + Kbb*phi_b = 0
    assert np.isclose(K[1, 0] * phi[0, 0] + K[1, 1] * phi[1, 0], 0.0)
