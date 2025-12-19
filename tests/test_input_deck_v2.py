from pathlib import Path

from dc_solver.io.abaqus_inp import parse_inp
from tools.inp_normalize import find_unused_amplitudes


def test_parse_portal_frame_v2_includes() -> None:
    data = parse_inp("examples/portal_frame_v2/main.inp")
    assert len(data.part.nodes) == 19
    assert len(data.part.elements) == 18
    assert "NS_BASE" in data.part.nsets
    assert "ES_BEAM" in data.part.elsets


def test_unused_amplitude_detection_job1() -> None:
    unused = find_unused_amplitudes("examples/Job-1.inp")
    assert "Amp-1" in unused
