from __future__ import annotations

from app.services.export_service import export_analysis_bundle_json
from app.services.frame_service import build_frame_model
from app.services.modal_service import run_modal_case
from app.services.section_service import build_section
from app.services.session_schema import AnalysisSettings, FrameInput, SectionSelection


def main() -> None:
    section = SectionSelection(material="steel", family="IPE", name="IPE100")
    _section_obj = build_section(section)

    frame = FrameInput(width=5.0, height=3.0, n_beam=2, mass_total=1_000.0)
    model = build_frame_model(frame)

    modal = run_modal_case(model, AnalysisSettings(n_modes=1))

    bundle_json = export_analysis_bundle_json(
        section_selection=section,
        frame_input=frame,
        modal_result=modal,
        dynamic_result=None,
    )
    print(bundle_json[:500])


if __name__ == "__main__":
    main()
