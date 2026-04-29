from __future__ import annotations

import json

import numpy as np

from app.services.export_service import (
    export_analysis_bundle,
    export_analysis_bundle_json,
    to_jsonable,
)
from app.services.session_schema import FrameInput, SectionSelection


def test_to_jsonable_numpy_arrays_and_scalars() -> None:
    payload = {
        "arr": np.array([1.0, 2.0, 3.0]),
        "scalar": np.float64(2.5),
        "nested": {"i": np.int64(3)},
    }
    out = to_jsonable(payload)
    assert out["arr"] == [1.0, 2.0, 3.0]
    assert isinstance(out["scalar"], float)
    assert out["nested"]["i"] == 3


def test_export_analysis_bundle_minimal_json_roundtrip() -> None:
    section = SectionSelection(material="steel", family="IPE", name="IPE100")
    frame = FrameInput(width=5.0, height=3.0, n_beam=2)

    bundle = export_analysis_bundle(section, frame)
    assert bundle["section_selection"]["name"] == "IPE100"
    assert bundle["frame_input"]["width"] == 5.0
    assert bundle["modal_result"] is None
    assert bundle["dynamic_result"] is None

    text = export_analysis_bundle_json(section, frame)
    parsed = json.loads(text)
    assert parsed["section_selection"]["family"] == "IPE"
    assert parsed["frame_input"]["height"] == 3.0
