from __future__ import annotations


def test_import_new_components() -> None:
    from app.components.dynamic_viewer import (
        render_dynamic_charts,
        render_dynamic_settings,
        render_dynamic_summary,
        render_ground_motion_settings,
    )
    from app.components.modal_viewer import (
        render_modal_settings,
        render_modal_summary,
        render_mode_shape_chart,
        render_spectrum_settings,
    )
    from app.components.result_cards import metric_card, render_analysis_warnings

    assert callable(render_modal_settings)
    assert callable(render_modal_summary)
    assert callable(render_mode_shape_chart)
    assert callable(render_spectrum_settings)
    assert callable(render_ground_motion_settings)
    assert callable(render_dynamic_settings)
    assert callable(render_dynamic_summary)
    assert callable(render_dynamic_charts)
    assert callable(metric_card)
    assert callable(render_analysis_warnings)


def test_import_new_pages() -> None:
    import importlib

    for module_name in ["app.pages.03_Modal", "app.pages.04_Dynamic", "app.pages.05_Results"]:
        module = importlib.import_module(module_name)
        assert module is not None
