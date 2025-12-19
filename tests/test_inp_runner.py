import numpy as np
from numpy.testing import assert_allclose

from dinamica_computacional.core.analysis import run_analysis
from dinamica_computacional.io.abaqus_like import read_inp


def _run_input(path: str):
    model, plan = read_inp(path)
    results = run_analysis(model, plan)
    return model, results


def test_sdof_input_runs(tmp_path):
    model, results = _run_input("tests/inputs/sdof.inp")
    step = results.dynamic_steps["DYNAMIC"]
    assert step["u"].shape[1] == model.ndof()
    assert step["u"].shape[0] == step["t"].size


def test_cantilever_input_static():
    model, results = _run_input("tests/inputs/cantilever.inp")
    u = results.static_steps["STATIC"]["u"]
    assert u.shape[0] == model.ndof()
    assert np.isfinite(u).all()


def test_portal_inputs_equilibrium():
    for path in ("tests/inputs/portal_linear.inp", "tests/inputs/portal_nlgeom.inp"):
        model, results = _run_input(path)
        u = results.static_steps["STATIC"]["u"]
        nd = model.ndof()
        assert u.shape == (nd,)

        load = model.load_const.copy()
        K = np.zeros((nd, nd))
        R = np.zeros(nd)
        for e in model.elements:
            dofs, k_g, f_g, _ = e.stiffness_and_force_global(u)
            for a, ia in enumerate(dofs):
                R[ia] += f_g[a]
                for b, ib in enumerate(dofs):
                    K[ia, ib] += k_g[a, b]
        residual = load - R
        assert_allclose(residual[model.free_dofs()], 0.0, atol=1e-6)
