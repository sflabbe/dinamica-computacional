from __future__ import annotations

import re
from datetime import datetime

from dc_solver.reporting import (
    AbaqusLikeReporter,
    IncrementEnd,
    IncrementStart,
    InputEcho,
    IterationReport,
    JobEnd,
    JobStart,
    StepEnd,
    StepStart,
    Warning,
)
from dc_solver.reporting.utils_format import format_sta_row


class FakeClock:
    def __init__(self, dt: datetime) -> None:
        self._dt = dt

    def __call__(self) -> datetime:
        return self._dt


def test_sta_row_marks_cutback() -> None:
    row = format_sta_row(
        step=1,
        inc=2,
        attempt=1,
        converged=False,
        severe_iters=0,
        equil_iters=3,
        total_iters=3,
        total_time=1.0,
        step_time=0.5,
        inc_time=0.5,
    )
    assert "1U" in row


def test_msg_contains_key_milestones(tmp_path) -> None:
    clock = FakeClock(datetime(2025, 12, 19, 10, 30, 0))
    reporter = AbaqusLikeReporter(tmp_path, "Job-1", clock=clock)
    reporter.on_event(
        JobStart(
            job_name="Job-1",
            start_dt=clock(),
            solver_version="dev",
            cwd=str(tmp_path),
            output_dir=str(tmp_path),
        )
    )
    reporter.on_event(
        StepStart(
            step_id=1,
            step_name="STEP-1",
            procedure="STATIC",
            total_time=0.0,
            nlgeom=False,
            dt0=1.0,
            dtmin=1.0,
            dtmax=1.0,
            max_increments=1,
        )
    )
    reporter.on_event(
        IncrementStart(
            step_id=1,
            inc=1,
            attempt=1,
            dt=1.0,
            step_time=0.0,
            total_time=0.0,
            is_cutback_attempt=False,
        )
    )
    reporter.on_event(
        IterationReport(
            step_id=1,
            inc=1,
            attempt=1,
            it=1,
            residual_norm=1.0,
            residual_max=1.0,
            residual_dof=3,
            residual_node=None,
            residual_component_label="FORCE",
            correction_norm=0.1,
            correction_max=0.1,
            converged_force=False,
            converged_moment=False,
            note=None,
        )
    )
    reporter.on_event(
        IterationReport(
            step_id=1,
            inc=1,
            attempt=1,
            it=2,
            residual_norm=1e-8,
            residual_max=1e-8,
            residual_dof=3,
            residual_node=None,
            residual_component_label="FORCE",
            correction_norm=0.0,
            correction_max=0.0,
            converged_force=True,
            converged_moment=True,
            note=None,
        )
    )
    reporter.on_event(
        IncrementEnd(
            step_id=1,
            inc=1,
            attempt=1,
            converged=True,
            n_equil_iters=2,
            n_severe_iters=0,
            dt_completed=1.0,
            step_fraction=1.0,
            step_time_completed=1.0,
            total_time_completed=1.0,
        )
    )
    reporter.on_event(StepEnd(step_id=1, step_time_completed=1.0, total_time_completed=1.0))
    reporter.on_event(
        JobEnd(
            success=True,
            end_dt=clock(),
            cpu_user_s=0.1,
            cpu_sys_s=0.0,
            wall_s=0.2,
            warnings_count=0,
            errors_count=0,
            totals={},
        )
    )
    reporter.close()

    msg_text = (tmp_path / "Job-1.msg").read_text(encoding="utf-8")
    assert re.search(r"STEP 1", msg_text)
    assert re.search(r"INCREMENT 1 STARTS", msg_text)
    assert re.search(r"CONVERGED", msg_text)
    assert re.search(r"ITERATION SUMMARY", msg_text)
    assert re.search(r"ANALYSIS SUMMARY", msg_text)


def test_dat_counts_warnings(tmp_path) -> None:
    clock = FakeClock(datetime(2025, 12, 19, 10, 30, 0))
    reporter = AbaqusLikeReporter(tmp_path, "Job-2", clock=clock)
    reporter.on_event(
        JobStart(
            job_name="Job-2",
            start_dt=clock(),
            solver_version="dev",
            cwd=str(tmp_path),
            output_dir=str(tmp_path),
        )
    )
    reporter.on_event(InputEcho(["*NODE", "1, 0, 0"]))
    reporter.on_event(Warning("First warning", phase="INPUT"))
    reporter.on_event(Warning("Second warning", phase="ANALYSIS"))
    reporter.on_event(
        JobEnd(
            success=True,
            end_dt=clock(),
            cpu_user_s=0.1,
            cpu_sys_s=0.0,
            wall_s=0.2,
            warnings_count=2,
            errors_count=0,
            totals={},
        )
    )
    reporter.close()

    dat_text = (tmp_path / "Job-2.dat").read_text(encoding="utf-8")
    assert "OPTIONS BEING PROCESSED" in dat_text
    assert "ANALYSIS COMPLETE WITH 2 WARNING MESSAGES ON THE DAT FILE" in dat_text
