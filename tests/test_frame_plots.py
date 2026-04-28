from pathlib import Path

import pytest

from dc_solver.post.frame_plots import (
    TypicalBalconyFrameSpec,
    plot_typical_balcony_frame,
    typical_balcony_frame_segments,
)


def test_typical_balcony_frame_segments_count_and_validation():
    spec = TypicalBalconyFrameSpec(floors=3, bay_width_m=4.0, story_height_m=3.0, balcony_depth_m=1.5)
    segments = typical_balcony_frame_segments(spec)
    assert len(segments) == 2 + 2 * 3
    assert segments[0][0] == (0.0, 0.0)
    assert segments[-1][1] == (5.5, 9.0)
    with pytest.raises(ValueError):
        typical_balcony_frame_segments(TypicalBalconyFrameSpec(floors=0))


def test_plot_typical_balcony_frame_writes_png(tmp_path: Path):
    out = tmp_path / "frame.png"
    written = plot_typical_balcony_frame(out, floors=2, show_node_ids=True)
    assert Path(written).exists()
    assert Path(written).stat().st_size > 0
