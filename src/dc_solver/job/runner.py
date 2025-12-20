"""JobRunner context manager for orchestrating analysis runs with full reporting."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Set

from dc_solver.job.file_tracker import snapshot_files, compute_new_files
from dc_solver.job.journal import JournalWriter
from dc_solver.job.console import print_job_header, print_job_footer
from dc_solver.job.flops import build_flops_report
from dc_solver.reporting.run_info import build_run_info, write_run_info
from dc_solver.reporting.abaqus_like import AbaqusLikeReporter, AbaqusLikeConfig


class JobRunner:
    """Context manager for running computational jobs with full Abaqus-style reporting.

    Handles:
    - Output directory setup
    - File tracking (snapshot before/after)
    - Journal logging
    - Abaqus-like .msg/.sta/.dat files
    - Run info (JSON + TXT)
    - FLOPs estimation
    - Console output (JOB START/END blocks)
    - Exception handling and error reporting

    Examples
    --------
    >>> with JobRunner(job_name="problema2", output_dir="outputs/problema2/run1") as job:
    ...     # Run your analysis
    ...     job.set_analysis_params(ndof=300, n_steps=10000, integrator="explicit")
    ...     # ... analysis code ...
    ...     job.mark_success()
    """

    def __init__(
        self,
        job_name: str,
        output_dir: str | Path,
        meta: Optional[Dict[str, Any]] = None,
        print_header: bool = True,
    ):
        """Initialize JobRunner.

        Parameters
        ----------
        job_name : str
            Name of the job (used for .msg/.sta/.dat files)
        output_dir : str | Path
            Output directory path
        meta : Optional[Dict[str, Any]]
            Metadata to include in run info
        print_header : bool, default=True
            Whether to print job header to console
        """
        self.job_name = job_name
        self.output_dir = Path(output_dir)
        self.meta = meta or {}
        self.print_header_flag = print_header

        # State
        self._success = False
        self._start_time = 0.0
        self._end_time = 0.0
        self._cpu_start = 0.0
        self._cpu_end = 0.0
        self._files_before: Set[Path] = set()
        self._files_after: Set[Path] = set()

        # Analysis parameters (for FLOPs estimation)
        self._ndof = 0
        self._n_steps = 0
        self._integrator = "unknown"
        self._avg_iterations = 1.0
        self._n_elements = None

        # Components
        self.journal: Optional[JournalWriter] = None
        self.reporter: Optional[AbaqusLikeReporter] = None

    def __enter__(self) -> JobRunner:
        """Enter context: setup output dir, start tracking."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot files before
        self._files_before = snapshot_files(self.output_dir, recursive=True)

        # Start timing
        self._start_time = time.time()
        self._cpu_start = time.process_time()

        # Open journal
        self.journal = JournalWriter(self.output_dir / "journal.log")
        self.journal.__enter__()
        self.journal.write(f"JOB START: {self.job_name}")

        # Open Abaqus-like reporters
        cfg = AbaqusLikeConfig(project_name="DinamicaComputacional")
        self.reporter = AbaqusLikeReporter(
            basepath=self.output_dir,
            job_name=self.job_name,
            clock=datetime.now,
            cfg=cfg,
        )

        # Fire JobStart event
        from dc_solver.reporting.events import JobStart
        job_start_event = JobStart(
            job_name=self.job_name,
            start_dt=datetime.now(),
            solver_version="dc_solver-0.1.0",
            cwd=str(Path.cwd()),
            output_dir=str(self.output_dir),
        )
        self.reporter.on_event(job_start_event)

        # Print header to console
        if self.print_header_flag:
            meta_keys = ", ".join(self.meta.keys()) if self.meta else None
            print_job_header(
                job_name=self.job_name,
                start_dt=datetime.now().isoformat(),
                output_dir=str(self.output_dir),
                meta_keys=meta_keys,
            )
            self.journal.write(f"  output_dir={self.output_dir}")
            if meta_keys:
                self.journal.write(f"  meta_keys={meta_keys}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: finalize reporting, close files."""
        # Stop timing
        self._end_time = time.time()
        self._cpu_end = time.process_time()

        wall_s = self._end_time - self._start_time
        cpu_s = self._cpu_end - self._cpu_start

        # If exception occurred, mark as failed
        if exc_type is not None:
            self._success = False
            # Log exception to journal and msg file
            tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            if self.journal:
                self.journal.write(f"EXCEPTION: {exc_type.__name__}: {exc_val}")
                self.journal.write_raw(tb_str)
            # Write to msg file if reporter exists
            if self.reporter and hasattr(self.reporter, "_msg"):
                self.reporter._msg._file.write("\n" + "=" * 80 + "\n")
                self.reporter._msg._file.write("EXCEPTION OCCURRED:\n")
                self.reporter._msg._file.write(tb_str)
                self.reporter._msg._file.write("=" * 80 + "\n")

        # Snapshot files after
        self._files_after = snapshot_files(self.output_dir, recursive=True)
        new_files = compute_new_files(
            self.output_dir,
            self._files_before,
            recursive=True,
        )

        # Compute FLOPs
        flops_report = build_flops_report(
            ndof=self._ndof,
            n_steps=self._n_steps,
            integrator=self._integrator,
            avg_iterations=self._avg_iterations,
            wall_seconds=wall_s,
            n_elements=self._n_elements,
        )

        # Fire JobEnd event
        if self.reporter:
            from dc_solver.reporting.events import JobEnd
            job_end_event = JobEnd(
                success=self._success,
                end_dt=datetime.now(),
                cpu_user_s=cpu_s,
                cpu_sys_s=0.0,  # Not tracked separately
                wall_s=wall_s,
                warnings_count=self.reporter.warnings_count,
                errors_count=self.reporter.errors_count,
                totals={
                    "flops_est": flops_report["flops_est"],
                    "gflops_rate": flops_report["gflops_rate"],
                    "new_files_count": len(new_files),
                },
            )
            self.reporter.on_event(job_end_event)
            self.reporter.close()

        # Print footer to console
        if self.print_header_flag:
            print_job_footer(
                job_name=self.job_name,
                status="OK" if self._success else "FAILED",
                end_dt=datetime.now().isoformat(),
                wall_clock=wall_s,
                cpu_time=cpu_s,
                flops_est=flops_report["flops_est"],
                gflops_rate=flops_report["gflops_rate"],
                output_dir=str(self.output_dir),
                new_files_count=len(new_files),
                new_files=[str(f) for f in new_files],
            )

        # Write run info
        run_info = build_run_info(
            job=self.job_name,
            output_dir=str(self.output_dir),
            meta=self.meta,
        )
        run_info.update({
            "success": self._success,
            "wall_s": wall_s,
            "cpu_s": cpu_s,
            "flops_est": flops_report["flops_est"],
            "gflops_rate": flops_report["gflops_rate"],
            "ndof": self._ndof,
            "n_steps": self._n_steps,
            "integrator": self._integrator,
            "new_files_count": len(new_files),
            "new_files": [str(f) for f in new_files],
        })
        if self.reporter:
            run_info["warnings_count"] = self.reporter.warnings_count
            run_info["errors_count"] = self.reporter.errors_count

        write_run_info(
            self.output_dir,
            base_name=f"{self.job_name}_runinfo",
            info=run_info,
        )

        # Close journal
        if self.journal:
            self.journal.write(f"JOB END: {self.job_name} ({'OK' if self._success else 'FAILED'})")
            self.journal.write(f"  wall_s={wall_s:.3f} cpu_s={cpu_s:.3f}")
            self.journal.write(f"  new_files={len(new_files)}")
            self.journal.__exit__(None, None, None)

        # Do not suppress exception
        return False

    def set_analysis_params(
        self,
        ndof: int,
        n_steps: int,
        integrator: str,
        avg_iterations: float = 1.0,
        n_elements: Optional[int] = None,
    ) -> None:
        """Set analysis parameters for FLOPs estimation.

        Parameters
        ----------
        ndof : int
            Number of degrees of freedom
        n_steps : int
            Number of time steps
        integrator : str
            Integrator name (e.g., 'explicit', 'implicit', 'newmark')
        avg_iterations : float, default=1.0
            Average iterations per step (for implicit)
        n_elements : Optional[int]
            Number of elements
        """
        self._ndof = ndof
        self._n_steps = n_steps
        self._integrator = integrator
        self._avg_iterations = avg_iterations
        self._n_elements = n_elements

        if self.journal:
            self.journal.write(f"Analysis params: ndof={ndof} n_steps={n_steps} integrator={integrator}")

    def mark_success(self) -> None:
        """Mark the job as successful."""
        self._success = True
        if self.journal:
            self.journal.write("Job marked as SUCCESS")

    def mark_failure(self, reason: str = "") -> None:
        """Mark the job as failed.

        Parameters
        ----------
        reason : str, default=""
            Reason for failure
        """
        self._success = False
        if self.journal:
            self.journal.write(f"Job marked as FAILED: {reason}")

    def log(self, message: str) -> None:
        """Write a message to the journal.

        Parameters
        ----------
        message : str
            Message to log
        """
        if self.journal:
            self.journal.write(message)
