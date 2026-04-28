import numpy as np
import pytest
from numpy.testing import assert_allclose

from dc_solver.io.abaqus_inp import parse_inp, build_model, apply_gravity, amplitude_series
from dc_solver.static.newton import solve_static_newton
from dc_solver.integrators.hht_alpha import hht_alpha_newton


def _run_static_step(path: str, nlgeom: bool) -> np.ndarray:
    data = parse_inp(path)
    model = build_model(data, nlgeom=nlgeom)
    gravity = None
    for step in data.steps:
        if step.gravity is not None:
            gravity = step.gravity
            break
    if gravity is not None:
        apply_gravity(model, data, gravity)
    u = solve_static_newton(model, model.load_const)
    return u


def test_job1_trimmed_parses_and_static_converges():
    data = parse_inp("tests/inputs/job1_trimmed.inp")
    assert len(data.part.nodes) == 6
    assert len(data.part.elements) == 5

    u = _run_static_step("tests/inputs/job1_trimmed.inp", nlgeom=True)
    assert np.isfinite(u).all()


@pytest.mark.slow
def test_portal_6seg_runs_static_and_dynamic():
    data = parse_inp("examples/abaqus_like/portal_6seg.inp")
    step_static = data.steps[0]
    model_static = build_model(data, nlgeom=step_static.nlgeom)
    if step_static.gravity is not None:
        apply_gravity(model_static, data, step_static.gravity)
    u = solve_static_newton(model_static, model_static.load_const)
    assert np.isfinite(u).all()

    step_dyn = data.steps[1]
    model_dyn = build_model(data, nlgeom=step_dyn.nlgeom)
    if step_static.gravity is not None:
        apply_gravity(model_dyn, data, step_static.gravity)
    t = np.arange(0.0, step_dyn.time_period + 1e-12, step_dyn.dt)
    amp = amplitude_series(data.amplitudes["AMP1"], step_dyn.dt, step_dyn.time_period)
    ag = amp * step_dyn.accel_bc[2]
    out = hht_alpha_newton(
        model_dyn,
        t,
        ag,
        drift_height=3.0,
        base_nodes=(0, 7),
        drift_nodes=(6, 13),
        alpha=-0.05,
    )
    assert out["u"].shape[0] == t.size
    assert out["u"].shape[1] == model_dyn.ndof()
    assert_allclose(out["u"][0], u, atol=1e-6)
