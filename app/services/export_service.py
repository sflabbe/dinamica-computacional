from __future__ import annotations

import json

from dc_solver.reporting.run_info import to_jsonable as _to_jsonable


def to_jsonable(obj):
    """Convert app/core objects to JSON-serializable Python primitives."""
    return _to_jsonable(obj)


def export_analysis_bundle(
    section_selection,
    frame_input,
    modal_result=None,
    dynamic_result=None,
) -> dict:
    """Build a traceable in-memory export bundle for app analysis outputs."""
    bundle = {
        "section_selection": to_jsonable(section_selection),
        "frame_input": to_jsonable(frame_input),
        "modal_result": to_jsonable(modal_result),
        "dynamic_result": to_jsonable(dynamic_result),
    }
    return bundle


def export_analysis_bundle_json(*args, **kwargs) -> str:
    """Serialize export_analysis_bundle() result to JSON text."""
    return json.dumps(export_analysis_bundle(*args, **kwargs), ensure_ascii=False, indent=2)
