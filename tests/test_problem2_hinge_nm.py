from __future__ import annotations

import numpy as np

from plastic_hinge import PlasticHingeNM
from problems.problema2_secciones_nm import make_section_S1, compute_interaction_polygon


def test_problem2_hinge_nm_random_steps_inside():
    section = make_section_S1()
    surface = compute_interaction_polygon(section, symmetric_M=True, symmetric_N=True, n=40)

    Lp = 0.5 * section.h
    E = 30e9
    I = section.b * section.h**3 / 12.0
    KN = E * section.Ac / Lp
    KM = E * I / Lp
    hinge = PlasticHingeNM(surface=surface, K=np.diag([KN, KM]))

    rng = np.random.default_rng(123)
    s = np.zeros(2)
    for _ in range(20):
        dq = rng.normal(scale=[1e-5, 5e-4], size=2)
        info = hinge.update(dq)
        s = info["s"]
        dq_p_inc = info["dq_p_inc"]
        assert surface.is_inside(s, tol=1e-8)
        dWp = float(np.dot(s, dq_p_inc))
        assert dWp >= -1e-12
