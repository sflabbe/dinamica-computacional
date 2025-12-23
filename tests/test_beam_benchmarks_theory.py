import numpy as np
from numpy.testing import assert_allclose

from dc_solver.io.abaqus_inp import parse_inp, build_model, apply_cloads
from dc_solver.static.newton import solve_static_newton


def _node_map_from_part(data):
    return {nid: idx for idx, (nid, _) in enumerate(sorted(data.part.nodes.items()))}


def _run_static(path: str):
    data = parse_inp(path)
    step = data.steps[0]
    model = build_model(data, nlgeom=step.nlgeom)
    # Use Euler-Bernoulli theory for analytical comparison
    for beam in model.beams:
        beam.beam_theory = "euler"
    if step.cloads:
        apply_cloads(model, data, step)
    u = solve_static_newton(model, model.load_const)
    return data, model, u


def test_cantilever_tip_load_matches_theory():
    data, model, u = _run_static("examples/abaqus_like/beam_cantilever_tipload.inp")
    node_map = _node_map_from_part(data)
    tip_idx = node_map[2]
    tip_node = model.nodes[tip_idx]
    uy_dof = tip_node.dof_u[1]
    th_dof = tip_node.dof_th

    xs = [x for x, _ in data.part.nodes.values()]
    L = max(xs) - min(xs)
    E = data.material.E
    I = model.beams[0].I
    P = 10000.0

    delta = P * L**3 / (3.0 * E * I)
    theta = P * L**2 / (2.0 * E * I)

    assert_allclose(u[uy_dof], -delta, rtol=1e-6, atol=1e-9)
    assert_allclose(u[th_dof], -theta, rtol=1e-6, atol=1e-9)


def test_simply_supported_midspan_load_matches_theory():
    data, model, u = _run_static("examples/abaqus_like/beam_simply_supported_midload.inp")
    node_map = _node_map_from_part(data)
    mid_idx = node_map[2]
    mid_node = model.nodes[mid_idx]
    uy_dof = mid_node.dof_u[1]

    xs = [x for x, _ in data.part.nodes.values()]
    L = max(xs) - min(xs)
    E = data.material.E
    I = model.beams[0].I
    P = 10000.0

    delta = P * L**3 / (48.0 * E * I)

    assert_allclose(u[uy_dof], -delta, rtol=1e-6, atol=1e-9)
