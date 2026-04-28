"""Small, dependency-light frame plotting helpers for downstream tools.

These helpers intentionally do not require a full FE `Model`.  They are meant
for report sketches and orchestrator repos (for example `balkon-automation`)
that need a deterministic structural frame drawing while the actual calculation
lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TypicalBalconyFrameSpec:
    floors: int = 2
    bay_width_m: float = 4.0
    story_height_m: float = 3.0
    balcony_depth_m: float = 1.5
    show_node_ids: bool = False
    title: str = "Typischer Balkonrahmen"

    def validate(self) -> None:
        if self.floors < 1:
            raise ValueError("floors must be >= 1")
        if self.bay_width_m <= 0.0:
            raise ValueError("bay_width_m must be > 0")
        if self.story_height_m <= 0.0:
            raise ValueError("story_height_m must be > 0")
        if self.balcony_depth_m < 0.0:
            raise ValueError("balcony_depth_m must be >= 0")


def typical_balcony_frame_segments(spec: TypicalBalconyFrameSpec) -> list[tuple[tuple[float, float], tuple[float, float], str]]:
    """Return line segments for a two-column frame with balcony cantilevers."""
    spec.validate()
    W = float(spec.bay_width_m)
    H = float(spec.story_height_m)
    T = float(spec.balcony_depth_m)
    n = int(spec.floors)
    segments: list[tuple[tuple[float, float], tuple[float, float], str]] = []

    # columns
    segments.append(((0.0, 0.0), (0.0, n * H), "Stütze links"))
    segments.append(((W, 0.0), (W, n * H), "Stütze rechts"))

    # floors and balcony cantilevers
    for i in range(1, n + 1):
        z = i * H
        segments.append(((0.0, z), (W, z), f"Riegel Geschoss {i}"))
        if T > 0.0:
            segments.append(((W, z), (W + T, z), f"Kragarm Balkon {i}"))
    return segments


def plot_typical_balcony_frame(
    output_path: str | Path,
    *,
    floors: int = 2,
    bay_width_m: float = 4.0,
    story_height_m: float = 3.0,
    balcony_depth_m: float = 1.5,
    show_node_ids: bool = False,
    title: str = "Typischer Balkonrahmen",
) -> str:
    """Render a deterministic frame sketch and return the written path."""
    spec = TypicalBalconyFrameSpec(
        floors=floors,
        bay_width_m=bay_width_m,
        story_height_m=story_height_m,
        balcony_depth_m=balcony_depth_m,
        show_node_ids=show_node_ids,
        title=title,
    )
    segments = typical_balcony_frame_segments(spec)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for idx, ((x0, y0), (x1, y1), label) in enumerate(segments, start=1):
        ax.plot([x0, x1], [y0, y1], linewidth=2.0)
        if show_node_ids:
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0, str(idx), fontsize=8)

    # supports
    ax.scatter([0.0, spec.bay_width_m], [0.0, 0.0], marker="^", s=70)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, spec.bay_width_m + spec.balcony_depth_m + 0.4)
    ax.set_ylim(-0.3, spec.floors * spec.story_height_m + 0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out)
