"""Abaqus-like log writers for status, message, and data outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from dc_solver.reporting import events
from dc_solver.reporting.utils_format import format_date, format_float, format_sta_row, format_time, sta_header


@dataclass
class AbaqusLikeConfig:
    project_name: str = "DinamicaComputacional"


class StaWriter:
    def __init__(self, path: Path, clock: Callable[[], datetime], cfg: AbaqusLikeConfig) -> None:
        self._file = path.open("w", encoding="utf-8", buffering=1)
        self._clock = clock
        self._cfg = cfg

    def on_job_start(self, event: events.JobStart) -> None:
        date_str = format_date(event.start_dt)
        time_str = format_time(event.start_dt)
        self._file.write(f"{self._cfg.project_name} STATUS FILE -- DATE {date_str} TIME {time_str}\n")
        self._file.write(sta_header() + "\n")
        self._file.flush()

    def on_increment_end(self, event: events.IncrementEnd) -> None:
        line = format_sta_row(
            event.step_id,
            event.inc,
            event.attempt,
            event.converged,
            event.n_severe_iters,
            event.n_equil_iters,
            event.n_severe_iters + event.n_equil_iters,
            event.total_time_completed,
            event.step_time_completed,
            event.dt_completed,
        )
        self._file.write(line + "\n")
        self._file.flush()

    def on_job_end(self, event: events.JobEnd) -> None:
        if event.success:
            self._file.write("THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n")
        else:
            self._file.write("THE ANALYSIS HAS TERMINATED WITH ERRORS\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class MsgWriter:
    def __init__(self, path: Path, clock: Callable[[], datetime], cfg: AbaqusLikeConfig) -> None:
        self._file = path.open("w", encoding="utf-8", buffering=1)
        self._clock = clock
        self._cfg = cfg
        self._step_id: Optional[int] = None
        self._step_name: Optional[str] = None
        self._procedure: Optional[str] = None
        self._increments = 0
        self._cutbacks = 0
        self._iterations = 0

    def on_job_start(self, event: events.JobStart) -> None:
        date_str = format_date(event.start_dt)
        time_str = format_time(event.start_dt)
        self._file.write(f"{self._cfg.project_name} MESSAGE FILE -- DATE {date_str} TIME {time_str}\n")
        self._file.flush()

    def on_step_start(self, event: events.StepStart) -> None:
        self._step_id = event.step_id
        self._step_name = event.step_name
        self._procedure = event.procedure
        self._file.write("\n")
        self._file.write(f"STEP {event.step_id}: {event.step_name}\n")
        self._file.write(f"PROCEDURE TYPE: {event.procedure}\n")
        self._file.write(f"STEP TIME COMPLETED = {format_float(event.total_time)}\n")
        self._file.flush()

    def on_increment_start(self, event: events.IncrementStart) -> None:
        if event.is_cutback_attempt:
            self._cutbacks += 1
        self._file.write(
            "INCREMENT {inc} STARTS. ATTEMPT NUMBER {att}, TIME INCREMENT {dt}\n".format(
                inc=event.inc,
                att=event.attempt,
                dt=format_float(event.dt),
            )
        )
        self._file.flush()

    def on_iteration(self, event: events.IterationReport) -> None:
        self._iterations += 1
        status = "CONVERGED" if event.converged_force and event.converged_moment else "NOT CONVERGED"
        resid_label = event.residual_component_label or "RESIDUAL"
        self._file.write(
            " ITER {it}: {status} -- {label} NORM {rnorm}, MAX {rmax}, CORR NORM {cnorm}\n".format(
                it=event.it,
                status=status,
                label=resid_label,
                rnorm=format_float(event.residual_norm),
                rmax=format_float(event.residual_max),
                cnorm=format_float(event.correction_norm),
            )
        )
        if event.note:
            self._file.write(f"  NOTE: {event.note}\n")
        self._file.flush()

    def on_increment_end(self, event: events.IncrementEnd) -> None:
        self._increments += 1
        if not event.converged:
            self._file.write(" INCREMENT NOT CONVERGED -- CUTBACK\n")
        self._file.write(
            " ITERATION SUMMARY: EQUIL ITERS {equil}, SEVERE DISCON ITERS {sev}\n".format(
                equil=event.n_equil_iters,
                sev=event.n_severe_iters,
            )
        )
        self._file.write(
            " TIME INCREMENT COMPLETED = {dt}, FRACTION OF STEP COMPLETED = {frac}\n".format(
                dt=format_float(event.dt_completed),
                frac=format_float(event.step_fraction),
            )
        )
        self._file.write(
            " TOTAL TIME COMPLETED = {total}\n".format(
                total=format_float(event.total_time_completed),
            )
        )
        self._file.flush()

    def on_job_end(self, event: events.JobEnd) -> None:
        self._file.write("\nANALYSIS SUMMARY\n")
        self._file.write(f" TOTAL INCREMENTS = {self._increments}\n")
        self._file.write(f" TOTAL CUTBACKS = {self._cutbacks}\n")
        self._file.write(f" TOTAL ITERATIONS = {self._iterations}\n")
        self._file.write(f" TOTAL WARNINGS = {event.warnings_count}\n")
        self._file.write(f" TOTAL ERRORS = {event.errors_count}\n")
        self._file.write("\nJOB TIME SUMMARY\n")
        self._file.write(
            " CPU TIME (USER) = {user}s, CPU TIME (SYSTEM) = {sys}s, WALL CLOCK = {wall}s\n".format(
                user=format_float(event.cpu_user_s),
                sys=format_float(event.cpu_sys_s),
                wall=format_float(event.wall_s),
            )
        )
        self._file.write(
            " ANALYSIS STATUS: {status}\n".format(
                status="SUCCESS" if event.success else "FAILED",
            )
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class DatWriter:
    def __init__(self, path: Path, clock: Callable[[], datetime], cfg: AbaqusLikeConfig) -> None:
        self._file = path.open("w", encoding="utf-8", buffering=1)
        self._clock = clock
        self._cfg = cfg
        self._warnings = 0

    def on_job_start(self, event: events.JobStart) -> None:
        date_str = format_date(event.start_dt)
        time_str = format_time(event.start_dt)
        self._file.write(
            f"{self._cfg.project_name} DATA FILE -- DATE {date_str} TIME {time_str}\n"
        )
        self._file.write("OPTIONS BEING PROCESSED\n")
        self._file.flush()

    def on_input_echo(self, event: events.InputEcho) -> None:
        for line in event.lines:
            self._file.write(f" {line}\n")
        self._file.flush()

    def on_warning(self, event: events.Warning) -> None:
        self._warnings += 1
        location = ""
        if event.step_id is not None:
            location += f" STEP {event.step_id}"
        if event.inc is not None:
            location += f" INC {event.inc}"
        self._file.write(f"WARNING{location}: {event.message}\n")
        self._file.flush()

    def on_job_end(self, event: events.JobEnd) -> None:
        self._file.write(
            f"ANALYSIS COMPLETE WITH {event.warnings_count} WARNING MESSAGES ON THE DAT FILE\n"
        )
        self._file.write("JOB TIME SUMMARY\n")
        self._file.write(
            " CPU TIME (USER) = {user}s, CPU TIME (SYSTEM) = {sys}s, WALL CLOCK = {wall}s\n".format(
                user=format_float(event.cpu_user_s),
                sys=format_float(event.cpu_sys_s),
                wall=format_float(event.wall_s),
            )
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class AbaqusLikeReporter:
    def __init__(
        self,
        basepath: Path,
        job_name: str,
        clock: Callable[[], datetime],
        cfg: Optional[AbaqusLikeConfig] = None,
    ) -> None:
        self.basepath = basepath
        self.basepath.mkdir(parents=True, exist_ok=True)
        self.job_name = job_name
        self.clock = clock
        self.cfg = cfg or AbaqusLikeConfig()
        self._warnings = 0
        self._errors = 0

        self._sta = StaWriter(basepath / f"{job_name}.sta", clock, self.cfg)
        self._msg = MsgWriter(basepath / f"{job_name}.msg", clock, self.cfg)
        self._dat = DatWriter(basepath / f"{job_name}.dat", clock, self.cfg)

    @property
    def warnings_count(self) -> int:
        return self._warnings

    @property
    def errors_count(self) -> int:
        return self._errors

    def on_event(self, event: object) -> None:
        if isinstance(event, events.Warning):
            self._warnings += 1
            self._dat.on_warning(event)
            return
        if isinstance(event, events.Error):
            self._errors += 1
            return
        if isinstance(event, events.JobStart):
            self._sta.on_job_start(event)
            self._msg.on_job_start(event)
            self._dat.on_job_start(event)
            return
        if isinstance(event, events.InputEcho):
            self._dat.on_input_echo(event)
            return
        if isinstance(event, events.StepStart):
            self._msg.on_step_start(event)
            return
        if isinstance(event, events.IncrementStart):
            self._msg.on_increment_start(event)
            return
        if isinstance(event, events.IterationReport):
            self._msg.on_iteration(event)
            return
        if isinstance(event, events.IncrementEnd):
            self._sta.on_increment_end(event)
            self._msg.on_increment_end(event)
            return
        if isinstance(event, events.StepEnd):
            return
        if isinstance(event, events.JobEnd):
            self._sta.on_job_end(event)
            self._msg.on_job_end(event)
            self._dat.on_job_end(event)
            return

    def close(self) -> None:
        self._sta.close()
        self._msg.close()
        self._dat.close()


def build_input_echo(lines: Sequence[str]) -> events.InputEcho:
    return events.InputEcho(lines=lines)
