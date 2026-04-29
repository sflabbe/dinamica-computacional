from dc_solver.sections import load_steel_profile


def test_load_profile_ipe_200():
    sec = load_steel_profile("IPE", "IPE 200")
    assert sec.name == "IPE 200"
    assert sec.properties().A > 0
