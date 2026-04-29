from dc_solver.sections import AluminumRectTube


def test_aluminum_properties_positive():
    sec = AluminumRectTube(name="RHS", b=0.2, h=0.1, t=0.005)
    p = sec.properties()
    assert p.A > 0
    assert p.I_y > 0
    assert p.I_z > 0


def test_ec9_placeholder():
    sec = AluminumRectTube(name="RHS", b=0.2, h=0.1, t=0.005)
    out = sec.preclassify_ec9_placeholder()
    assert out["status"] == "manual_verification_required"
