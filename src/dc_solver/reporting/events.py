"""Event definitions for solver reporting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class JobStart:
    job_name: str
    start_dt: datetime
    solver_version: str
    cwd: str
    output_dir: str


@dataclass(frozen=True)
class JobEnd:
    success: bool
    end_dt: datetime
    cpu_user_s: float
    cpu_sys_s: float
    wall_s: float
    warnings_count: int
    errors_count: int
    totals: Dict[str, float]


@dataclass(frozen=True)
class StepStart:
    step_id: int
    step_name: str
    procedure: str
    total_time: float
    nlgeom: bool
    dt0: float
    dtmin: float
    dtmax: float
    max_increments: int
    hht_params: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class StepEnd:
    step_id: int
    step_time_completed: float
    total_time_completed: float


@dataclass(frozen=True)
class IncrementStart:
    step_id: int
    inc: int
    attempt: int
    dt: float
    step_time: float
    total_time: float
    is_cutback_attempt: bool


@dataclass(frozen=True)
class IterationReport:
    step_id: int
    inc: int
    attempt: int
    it: int
    residual_norm: float
    residual_max: float
    residual_dof: Optional[int]
    residual_node: Optional[int]
    residual_component_label: str
    correction_norm: float
    correction_max: float
    converged_force: bool
    converged_moment: bool
    note: Optional[str] = None


@dataclass(frozen=True)
class IncrementEnd:
    step_id: int
    inc: int
    attempt: int
    converged: bool
    n_equil_iters: int
    n_severe_iters: int
    dt_completed: float
    step_fraction: float
    step_time_completed: float
    total_time_completed: float


@dataclass(frozen=True)
class Warning:
    message: str
    phase: str
    step_id: Optional[int] = None
    inc: Optional[int] = None


@dataclass(frozen=True)
class Error:
    message: str
    step_id: Optional[int] = None
    inc: Optional[int] = None


@dataclass(frozen=True)
class InputEcho:
    lines: Sequence[str]
