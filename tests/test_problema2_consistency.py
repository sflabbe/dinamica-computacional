from __future__ import annotations

import numpy as np

from plastic_hinge import PlasticHingeNM
from problems.problema2_secciones_nm import make_section_S1, compute_interaction_polygon


def test_problema2_consistency_combined_history():
    section = make_section_S1()
    surface = compute_interaction_polygon(section, symmetric_M=True, symmetric_N=False, n=60)

    Lp = 0.5 * section.h
    E = 30e9
    I = section.b * section.h**3 / 12.0
    KN = E * section.Ac / Lp
    KM = E * I / Lp
    hinge = PlasticHingeNM(surface=surface, K=np.diag([KN, KM]), enable_substepping=True)

    t = np.linspace(0.0, 4.0 * np.pi, 200)
    eps_comb = 2.0e-4 * np.sin(t)
    th_comb = 0.006 * np.sin(2.0 * t + 0.3)
    q_hist = np.column_stack([eps_comb, th_comb])

    dq_hist = np.diff(q_hist, axis=0)
    max_violation = 0.0
    for dq in dq_hist:
        info = hinge.update(dq)
        s = info["s"]
        max_violation = max(max_violation, float(np.max(surface.A @ s - surface.b)))

    assert max_violation <= 1e-4  # Relaxed tolerance for return-mapping numerical error
