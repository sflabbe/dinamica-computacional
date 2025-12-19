from dc_solver.io.abaqus_inp import apply_cloads, build_model, parse_inp
from dc_solver.run import beam_benchmark_report
from dc_solver.static.newton import solve_static_newton


def _run_static(path: str):
    data = parse_inp(path)
    step = data.steps[0]
    model = build_model(data, nlgeom=step.nlgeom)
    if step.cloads:
        apply_cloads(model, data, step)
    u = solve_static_newton(model, model.load_const)
    return data, model, u


def test_beam_cantilever_reporting_keys():
    data, model, u = _run_static("examples/abaqus_like/beam_cantilever_tipload.inp")
    report = beam_benchmark_report(model, data, u, "cantilever")
    assert "uy_tip" in report
    assert "theta_tip" in report
    assert "theory_uy_tip" in report
    assert "theory_theta_tip" in report


def test_beam_simply_supported_reporting_keys():
    data, model, u = _run_static("examples/abaqus_like/beam_simply_supported_midload.inp")
    report = beam_benchmark_report(model, data, u, "simply_supported")
    assert "uy_mid" in report
    assert "theory_uy_mid" in report
