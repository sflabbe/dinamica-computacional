from dc_solver.sections import load_steel_profile


def test_cm2_to_m2_factor():
    sec = load_steel_profile("IPE", "IPE 200")
    assert abs(sec.A - 28.5e-4) < 1e-12


def test_cm4_to_m4_factor():
    sec = load_steel_profile("IPE", "IPE 200")
    assert abs(sec.I_y - 1940.0e-8) < 1e-16
