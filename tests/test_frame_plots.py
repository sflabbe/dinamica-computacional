from pathlib import Path

import matplotlib
import pytest

from dc_solver.post import (
    TypicalBalconyFrameSpec,
    plot_typical_balcony_frame,
    typical_balcony_frame_segments,
)


def test_typical_balcony_frame_segments_use_facade_grid():
    spec = TypicalBalconyFrameSpec(
        floors=3,
        bay_width_m=4.0,
        story_height_m=3.0,
        balcony_depth_m=1.5,
        facade_x_m=10.0,
    )

    segments = typical_balcony_frame_segments(spec)

    assert len(segments) == 2 + 2 * 3
    assert segments[0] == ((6.0, 0.0), (6.0, 9.0), "interior_column")
    assert segments[1] == ((10.0, 0.0), (10.0, 9.0), "facade_column")
    assert segments[-1] == ((10.0, 9.0), (11.5, 9.0), "balcony_3")


def test_typical_balcony_frame_segments_skip_zero_depth_balconies():
    spec = TypicalBalconyFrameSpec(
        floors=2,
        bay_width_m=5.0,
        story_height_m=2.8,
        balcony_depth_m=0.0,
        facade_x_m=0.0,
    )

    segments = typical_balcony_frame_segments(spec)

    assert len(segments) == 2 + 2
    assert [label for _, _, label in segments] == [
        "interior_column",
        "facade_column",
        "floor_1",
        "floor_2",
    ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"floors": 0}, "floors must be >= 1"),
        ({"bay_width_m": 0.0}, "bay_width_m must be > 0"),
        ({"story_height_m": -1.0}, "story_height_m must be > 0"),
        ({"balcony_depth_m": -0.1}, "balcony_depth_m must be >= 0"),
        ({"facade_x_m": float("nan")}, "facade_x_m must be finite"),
        ({"show_node_labels": "yes"}, "show_node_labels must be a bool"),
    ],
)
def test_typical_balcony_frame_spec_validation(kwargs, match):
    values = {
        "floors": 2,
        "bay_width_m": 4.0,
        "story_height_m": 3.0,
        "balcony_depth_m": 1.2,
        "facade_x_m": 0.0,
        "show_node_labels": False,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        typical_balcony_frame_segments(TypicalBalconyFrameSpec(**values))


def test_plot_typical_balcony_frame_writes_png_with_agg_backend(tmp_path: Path):
    out = tmp_path / "nested" / "frame.png"
    spec = TypicalBalconyFrameSpec(
        floors=4,
        bay_width_m=4.5,
        story_height_m=3.1,
        balcony_depth_m=1.7,
        facade_x_m=0.25,
        show_node_labels=True,
    )

    written = plot_typical_balcony_frame(spec, out)

    assert Path(written) == out
    assert out.exists()
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert out.stat().st_size > 0
    assert matplotlib.get_backend().lower() == "agg"


def test_plot_typical_balcony_frame_keeps_legacy_output_path_call(tmp_path: Path):
    out = tmp_path / "legacy.png"

    written = plot_typical_balcony_frame(out, floors=2, show_node_ids=True)

    assert Path(written).exists()
