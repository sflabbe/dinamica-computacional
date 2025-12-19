import numpy as np

from dinamica_computacional.integrators.hht_alpha import hht_alpha_newton

from tests.fixtures import assemble_full, build_sdof_column_model


def _energy_from_history(k: float, m: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 0.5 * m * v ** 2 + 0.5 * k * u ** 2


def test_energy_conservation_alpha_zero():
    model = build_sdof_column_model()
    nd = model.ndof()
    free = model.free_dofs()
    K_full, _ = assemble_full(model, np.zeros(nd))
    k = float(K_full[np.ix_(free, free)][0, 0])
    omega = 2.0 * np.pi
    m = k / (omega**2)
    model.mass_diag[:] = 0.0
    model.mass_diag[free[0]] = m

    dt = 0.002
    t = np.arange(0.0, 2.0 + 1e-12, dt)
    u0 = np.zeros(nd)
    u0[free[0]] = 1e-3

    out = hht_alpha_newton(
        model,
        t,
        ag=np.zeros_like(t),
        drift_height=max(nd.y for nd in model.nodes),
        alpha=0.0,
        u0=u0,
        v0=np.zeros(nd),
    )

    energy = _energy_from_history(k, m, out["u"][:, free[0]], out["v"][:, free[0]])
    rel_dev = np.max(np.abs(energy - energy[0])) / energy[0]
    assert rel_dev < 0.02


def test_energy_decay_with_numerical_damping():
    model = build_sdof_column_model()
    nd = model.ndof()
    free = model.free_dofs()
    K_full, _ = assemble_full(model, np.zeros(nd))
    k = float(K_full[np.ix_(free, free)][0, 0])
    omega = 2.0 * np.pi
    m = k / (omega**2)
    model.mass_diag[:] = 0.0
    model.mass_diag[free[0]] = m

    dt = 0.002
    t = np.arange(0.0, 2.0 + 1e-12, dt)
    u0 = np.zeros(nd)
    u0[free[0]] = 1e-3

    out = hht_alpha_newton(
        model,
        t,
        ag=np.zeros_like(t),
        drift_height=max(nd.y for nd in model.nodes),
        alpha=-0.05,
        u0=u0,
        v0=np.zeros(nd),
    )

    energy = _energy_from_history(k, m, out["u"][:, free[0]], out["v"][:, free[0]])
    assert energy[-1] <= energy[0]
