import numpy as np

from dc_solver.modal.modal_analysis import run_modal_analysis


class DummyNode:
    def __init__(self, ux, uy):
        self.dof_u = (ux, uy)


class DummyModel:
    def __init__(self):
        self.mass_diag = np.array([2.0, 3.0, 0.0])
        self.fixed_dofs = np.array([], dtype=int)
        self.nodes = [DummyNode(0, 1)]

    def ndof(self):
        return 3

    def free_dofs(self):
        return np.array([0, 1, 2])

    def assemble(self, u_trial, u_comm):
        K = np.array([[12.0, 0.0, -2.0], [0.0, 9.0, -3.0], [-2.0, -3.0, 5.0]])
        R = np.zeros(3)
        return K, R, {}


def test_modal_mass_normalization_and_cumulative_ratio():
    res = run_modal_analysis(DummyModel(), n_modes=2, direction="ux")
    M = np.diag([2.0, 3.0, 0.0])
    check = res.modes_full.T @ M @ res.modes_full
    assert np.allclose(check, np.eye(2), atol=1e-8)
    assert np.all(res.cumulative_mass_ratio >= 0.0)
    assert res.cumulative_mass_ratio[-1] <= 1.000001
