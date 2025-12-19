import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.integrators.hht_alpha import hht_alpha_newton

from tests.fixtures import assemble_full, build_sdof_column_model


def _setup_sdof(omega: float = 2.0 * np.pi):
    model = build_sdof_column_model()
    nd = model.ndof()
    free = model.free_dofs()
    K_full, _ = assemble_full(model, np.zeros(nd))
    k = float(K_full[np.ix_(free, free)][0, 0])
    m = k / (omega**2)
    model.mass_diag[:] = 0.0
    model.mass_diag[free[0]] = m
    model.C_diag[:] = 0.0
    return model, free[0], k, m


def test_hht_alpha_matches_sdof_cosine_response():
    omega = 2.0 * np.pi
    model, dof, k, m = _setup_sdof(omega=omega)

    dt = 0.002
    t = np.arange(0.0, 2.0 + 1e-12, dt)
    amp = 1e-3
    u0 = np.zeros(model.ndof())
    u0[dof] = amp

    out = hht_alpha_newton(
        model,
        t,
        ag=np.zeros_like(t),
        drift_height=max(nd.y for nd in model.nodes),
        alpha=0.0,
        u0=u0,
        v0=np.zeros(model.ndof()),
    )

    disp = out["u"][:, dof]
    expected = amp * np.cos(omega * t)
    rms = np.sqrt(np.mean((disp - expected) ** 2))
    assert rms < 1e-2

    energy = 0.5 * m * out["v"][:, dof] ** 2 + 0.5 * k * disp ** 2
    assert_allclose(energy, energy[0], rtol=2e-2)


def test_hht_alpha_numerical_damping_decreases_energy():
    model, dof, k, m = _setup_sdof()
    dt = 0.002
    t = np.arange(0.0, 2.0 + 1e-12, dt)
    u0 = np.zeros(model.ndof())
    u0[dof] = 1e-3

    out = hht_alpha_newton(
        model,
        t,
        ag=np.zeros_like(t),
        drift_height=max(nd.y for nd in model.nodes),
        alpha=-0.05,
        u0=u0,
        v0=np.zeros(model.ndof()),
    )

    disp = out["u"][:, dof]
    energy = 0.5 * m * out["v"][:, dof] ** 2 + 0.5 * k * disp ** 2
    mid = energy.size // 2
    assert np.max(energy[mid:]) <= np.max(energy[:mid]) * (1.0 + 1e-6)


def test_base_accel_equivalent_to_applied_load_history():
    model, dof, k, m = _setup_sdof()
    dt = 0.005
    t = np.arange(0.0, 1.0 + 1e-12, dt)
    ag = 0.3 * np.sin(2.0 * np.pi * t)

    out_ag = hht_alpha_newton(
        model,
        t,
        ag=ag,
        drift_height=max(nd.y for nd in model.nodes),
        alpha=0.0,
    )

    r = np.zeros(model.ndof())
    r[model.mass_diag > 0.0] = 1.0
    load_hist = -model.mass_diag[None, :] * r[None, :] * ag[:, None]

    out_load = hht_alpha_newton(
        model,
        t,
        ag=np.zeros_like(t),
        drift_height=max(nd.y for nd in model.nodes),
        alpha=0.0,
        load_hist=load_hist,
    )

    assert_allclose(out_ag["u"], out_load["u"], rtol=1e-6, atol=1e-8)
