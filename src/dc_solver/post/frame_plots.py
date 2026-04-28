"""Frame plotting helpers for downstream tools.

The functions in this module intentionally do not require a finite-element
``Model``.  They provide deterministic geometry sketches for report pipelines
that need a typical balcony frame plot before or outside a full analysis run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


Point2D = tuple[float, float]
FrameSegment = tuple[Point2D, Point2D, str]


@dataclass(frozen=True)
class TypicalBalconyFrameSpec:
    """Geometry for a typical two-column balcony frame sketch.

    ``facade_x_m`` is the facade column/grid line.  The interior column is one
    bay width to the left, and balconies extend in the positive x direction.
    """

    floors: int = 2
    bay_width_m: float = 4.0
    story_height_m: float = 3.0
    balcony_depth_m: float = 1.5
    facade_x_m: float = 0.0
    show_node_labels: bool = False
    title: str = "Typical balcony frame"

    def validate(self) -> None:
        if isinstance(self.floors, bool) or not isinstance(self.floors, Integral):
            raise ValueError("floors must be an integer")
        if self.floors < 1:
            raise ValueError("floors must be >= 1")
        _require_finite_positive("bay_width_m", self.bay_width_m)
        _require_finite_positive("story_height_m", self.story_height_m)
        _require_finite_non_negative("balcony_depth_m", self.balcony_depth_m)
        _require_finite("facade_x_m", self.facade_x_m)
        if not isinstance(self.show_node_labels, bool):
            raise ValueError("show_node_labels must be a bool")

    @property
    def show_node_ids(self) -> bool:
        """Backward-compatible alias for earlier callers."""
        return self.show_node_labels


def _require_finite(name: str, value: float) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")


def _require_finite_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _require_finite_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


def typical_balcony_frame_segments(spec: TypicalBalconyFrameSpec) -> list[FrameSegment]:
    """Return deterministic line segments for the typical balcony frame."""
    spec.validate()
    bay_width = float(spec.bay_width_m)
    story_height = float(spec.story_height_m)
    balcony_depth = float(spec.balcony_depth_m)
    facade_x = float(spec.facade_x_m)
    interior_x = facade_x - bay_width
    floors = int(spec.floors)
    top_y = floors * story_height

    segments: list[FrameSegment] = [
        ((interior_x, 0.0), (interior_x, top_y), "interior_column"),
        ((facade_x, 0.0), (facade_x, top_y), "facade_column"),
    ]

    for floor in range(1, floors + 1):
        y = floor * story_height
        segments.append(((interior_x, y), (facade_x, y), f"floor_{floor}"))
        if balcony_depth > 0.0:
            segments.append(((facade_x, y), (facade_x + balcony_depth, y), f"balcony_{floor}"))

    return segments


def _node_positions(spec: TypicalBalconyFrameSpec) -> list[Point2D]:
    bay_width = float(spec.bay_width_m)
    story_height = float(spec.story_height_m)
    balcony_depth = float(spec.balcony_depth_m)
    facade_x = float(spec.facade_x_m)
    interior_x = facade_x - bay_width

    nodes: list[Point2D] = []
    for level in range(0, int(spec.floors) + 1):
        y = level * story_height
        nodes.append((interior_x, y))
        nodes.append((facade_x, y))
        if level > 0 and balcony_depth > 0.0:
            nodes.append((facade_x + balcony_depth, y))
    return nodes


def plot_typical_balcony_frame(
    spec: TypicalBalconyFrameSpec | str | Path,
    output_path: str | Path | None = None,
    *,
    floors: int = 2,
    bay_width_m: float = 4.0,
    story_height_m: float = 3.0,
    balcony_depth_m: float = 1.5,
    facade_x_m: float = 0.0,
    show_node_labels: bool = False,
    show_node_ids: bool | None = None,
    title: str = "Typical balcony frame",
) -> str:
    """Render a PNG of a typical balcony frame and return the written path.

    The stable API is ``plot_typical_balcony_frame(spec, output_path)``.  The
    keyword arguments remain for older callers that passed ``output_path`` as
    the first positional argument.
    """
    frame_spec, out = _coerce_plot_arguments(
        spec=spec,
        output_path=output_path,
        floors=floors,
        bay_width_m=bay_width_m,
        story_height_m=story_height_m,
        balcony_depth_m=balcony_depth_m,
        facade_x_m=facade_x_m,
        show_node_labels=show_node_labels,
        show_node_ids=show_node_ids,
        title=title,
    )
    frame_spec.validate()

    segments = typical_balcony_frame_segments(frame_spec)
    nodes = _node_positions(frame_spec)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.ioff()
    fig, ax = plt.subplots(figsize=_figure_size(frame_spec))
    try:
        for (x0, y0), (x1, y1), label in segments:
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=_segment_color(label),
                linewidth=_segment_width(label),
                solid_capstyle="round",
                zorder=2,
            )

        if nodes:
            xs, ys = zip(*nodes)
            ax.scatter(xs, ys, s=22, color="0.16", zorder=3)

        if frame_spec.show_node_labels:
            _draw_node_labels(ax, nodes)

        ax.axvline(float(frame_spec.facade_x_m), color="0.55", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(frame_spec.title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.24)
        ax.margins(x=0.08, y=0.08)
        fig.tight_layout()
        fig.savefig(out, dpi=160, bbox_inches="tight", format="png")
    finally:
        plt.close(fig)

    return str(out)


def _coerce_plot_arguments(
    *,
    spec: TypicalBalconyFrameSpec | str | Path,
    output_path: str | Path | None,
    floors: int,
    bay_width_m: float,
    story_height_m: float,
    balcony_depth_m: float,
    facade_x_m: float,
    show_node_labels: bool,
    show_node_ids: bool | None,
    title: str,
) -> tuple[TypicalBalconyFrameSpec, Path]:
    if isinstance(spec, TypicalBalconyFrameSpec):
        if output_path is None:
            raise TypeError("output_path is required")
        return spec, Path(output_path)

    if output_path is not None:
        raise TypeError("legacy plotting calls must pass only one output path")

    labels = show_node_labels if show_node_ids is None else bool(show_node_ids)
    return (
        TypicalBalconyFrameSpec(
            floors=floors,
            bay_width_m=bay_width_m,
            story_height_m=story_height_m,
            balcony_depth_m=balcony_depth_m,
            facade_x_m=facade_x_m,
            show_node_labels=labels,
            title=title,
        ),
        Path(spec),
    )


def _figure_size(spec: TypicalBalconyFrameSpec) -> tuple[float, float]:
    width = max(5.2, min(8.0, 2.0 + 0.7 * (spec.bay_width_m + spec.balcony_depth_m)))
    height = max(4.0, min(8.5, 2.2 + 0.85 * spec.floors))
    return width, height


def _segment_color(label: str) -> str:
    if label.startswith("balcony_"):
        return "#bf5b17"
    if label.endswith("_column"):
        return "#1f4e79"
    return "#2f6f44"


def _segment_width(label: str) -> float:
    if label.startswith("balcony_"):
        return 2.0
    if label.endswith("_column"):
        return 2.4
    return 2.2


def _draw_node_labels(ax, nodes: list[Point2D]) -> None:
    if not nodes:
        return
    xs, ys = zip(*nodes)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    offset = 0.018 * span
    for idx, (x, y) in enumerate(nodes, start=1):
        ax.text(x + offset, y + offset, f"N{idx}", fontsize=7, color="0.22", zorder=4)
