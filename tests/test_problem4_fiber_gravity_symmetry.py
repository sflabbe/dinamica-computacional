from __future__ import annotations

import numpy as np

from dc_solver.utils.gravity import solve_gravity_only
from problems.problema4_portico import build_portal_beam_hinge


def test_problem4_fiber_gravity_symmetry():
    model, _ = build_portal_beam_hinge(beam_hinge="fiber", fiber_line_search=True)
    res = solve_gravity_only(model, line_search=True)

    assert bool(res["converged"]) is True
    assert abs(float(res["drift"])) < 1e-8

    roof_nodes = tuple(int(i) for i in res["roof_nodes"])
    roof_ux = [
        float(res["u"][model.nodes[i].dof_u[0]])
        for i in roof_nodes
    ]
    assert abs(float(np.mean(roof_ux))) < 1e-8
