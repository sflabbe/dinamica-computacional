from dc_solver.sections.base import SectionProperties


def test_plastic_moment_property():
    p = SectionProperties(
        name="x", A=0.01, I_y=1e-5, I_z=2e-5,
        W_el_y=1e-4, W_pl_y=1.2e-4, W_el_z=8e-5, W_pl_z=9e-5,
        i_y=0.03, i_z=0.04, E=210e9, fy=355e6, source="test"
    )
    assert p.M_pl_y == p.W_pl_y * p.fy
