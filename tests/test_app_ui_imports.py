from __future__ import annotations


def test_import_app_module() -> None:
    import app.main as app_module

    assert app_module is not None


def test_import_components() -> None:
    from app.components.frame_editor import render_frame_input, render_frame_preview, render_frame_summary
    from app.components.section_viewer import render_section_properties, render_section_selector

    assert callable(render_section_selector)
    assert callable(render_section_properties)
    assert callable(render_frame_input)
    assert callable(render_frame_summary)
    assert callable(render_frame_preview)


def test_import_services() -> None:
    import app.services as services

    assert "build_frame_model" in services.__all__
