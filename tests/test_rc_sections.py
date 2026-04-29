from dc_solver.sections import rc_gross_properties_rect


def test_rc_gross_rectangle_properties():
    b = 0.3
    h = 0.5
    p = rc_gross_properties_rect("RC", b=b, h=h, E_cm=30e9, fc=30e6)
    assert p.A == b * h
    assert p.I_y == b * h**3 / 12.0
