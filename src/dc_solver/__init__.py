"""dc_solver — 2D frame solver for structural dynamics."""

from __future__ import annotations

# FEM primitives
from dc_solver.fem.nodes import Node, DofManager
from dc_solver.fem.frame2d import FrameElementLinear2D
from dc_solver.fem.model import Model
from dc_solver.fem.utils import discretize_member

# Solvers
from dc_solver.static.newton import solve_static_newton
from dc_solver.utils.gravity import solve_gravity_only, GravityResult
from dc_solver.integrators import solve_dynamic

# Hinge models (most-used)
from dc_solver.hinges.models import (
    RotSpringElement,
    FiberRotSpringElement,
    SHMBeamHinge1D,
    FiberBeamHinge1D,
    ColumnHingeNMRot,
)

__version__ = "0.2.0"
__all__ = [
    "Node", "DofManager",
    "FrameElementLinear2D",
    "Model",
    "discretize_member",
    "solve_static_newton",
    "solve_gravity_only", "GravityResult",
    "solve_dynamic",
    "RotSpringElement", "FiberRotSpringElement",
    "SHMBeamHinge1D", "FiberBeamHinge1D", "ColumnHingeNMRot",
    "__version__",
]
