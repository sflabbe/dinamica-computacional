from dc_solver.examples.portal_frame import build_portal_beam_hinge


def test_portal_builder_is_importable():
    model, meta = build_portal_beam_hinge()
    assert model is not None
    assert isinstance(meta, dict)
