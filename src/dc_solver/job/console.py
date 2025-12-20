"""Console output utilities for pretty printing and progress tracking."""

from __future__ import annotations

from typing import Optional, Dict, Any


def print_job_header(
    job_name: str,
    start_dt: str,
    output_dir: str,
    meta_keys: Optional[str] = None,
) -> None:
    """Print job start header in Abaqus style.

    Parameters
    ----------
    job_name : str
        Name of the job
    start_dt : str
        Start datetime (ISO format)
    output_dir : str
        Output directory path
    meta_keys : Optional[str]
        Comma-separated meta keys
    """
    print("=" * 88)
    print(f"[dc_solver] JOB START: {job_name}")
    print(f"  start_dt   : {start_dt}")
    print(f"  output_dir : {output_dir}")
    if meta_keys:
        print(f"  meta keys  : {meta_keys}")
    print("=" * 88)


def print_job_footer(
    job_name: str,
    status: str,
    end_dt: str,
    wall_clock: float,
    cpu_time: float,
    flops_est: float,
    gflops_rate: float,
    output_dir: str,
    new_files_count: int,
    new_files: Optional[list] = None,
) -> None:
    """Print job end footer in Abaqus style.

    Parameters
    ----------
    job_name : str
        Name of the job
    status : str
        "OK" or "FAILED"
    end_dt : str
        End datetime (ISO format)
    wall_clock : float
        Wall clock time in seconds
    cpu_time : float
        CPU time in seconds
    flops_est : float
        Estimated FLOPs
    gflops_rate : float
        GFLOP/s rate
    output_dir : str
        Output directory
    new_files_count : int
        Number of new files created
    new_files : Optional[list]
        List of new file paths (relative)
    """
    print("-" * 88)
    print(f"[dc_solver] JOB END  : {job_name} ({status})")
    print(f"  end_dt     : {end_dt}")
    print(f"  wall_clock : {wall_clock:.3f} s")
    print(f"  cpu_time   : {cpu_time:.3f} s")
    print(f"  FLOPs est. : {flops_est:.3e}  (~{gflops_rate:.2f} GFLOP/s)")
    print(f"  output_dir : {output_dir}")
    print(f"  new files  : {new_files_count}")

    if new_files:
        # Print up to 20 files to avoid cluttering
        display_files = new_files[:20]
        for f in display_files:
            print(f"    - {f}")
        if len(new_files) > 20:
            print(f"    ... and {len(new_files) - 20} more")

    print("-" * 88)


def print_progress(
    step_idx: int,
    n_steps: int,
    t: float,
    dt: float,
    substep: int = 0,
    drift_peak: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Print incremental progress during analysis.

    Parameters
    ----------
    step_idx : int
        Current step index (0-based or 1-based)
    n_steps : int
        Total number of steps
    t : float
        Current time
    dt : float
        Current time increment
    substep : int, default=0
        Substep count (if substepping active)
    drift_peak : float, default=0.0
        Peak drift percentage
    extra : Optional[Dict[str, Any]]
        Extra metrics to display (e.g., energy residual)
    """
    pct = 100.0 * step_idx / max(1, n_steps)
    msg = (
        f"[dc_solver] PROGRESS: step={step_idx}/{n_steps} ({pct:.2f}%) "
        f"t={t:.4f} dt={dt:.3e} sub={substep} drift_peak={drift_peak:.2f}%"
    )
    if extra:
        extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
        msg += f" {extra_str}"
    print(msg)


def should_print_progress(
    step_idx: int,
    n_steps: int,
    print_every_steps: Optional[int] = None,
    print_every_pct: Optional[float] = None,
) -> bool:
    """Determine if progress should be printed at this step.

    Parameters
    ----------
    step_idx : int
        Current step index
    n_steps : int
        Total steps
    print_every_steps : Optional[int]
        Print every N steps (if provided)
    print_every_pct : Optional[float]
        Print every X% (if provided, e.g., 1.0 for every 1%)

    Returns
    -------
    bool
        True if progress should be printed
    """
    # Default: print every 5% or every 100 steps, whichever is less frequent
    if print_every_steps is None and print_every_pct is None:
        default_pct_steps = max(1, int(0.05 * n_steps))  # 5%
        default_fixed_steps = 100
        print_every_steps = min(default_pct_steps, default_fixed_steps)

    if print_every_steps is not None:
        if step_idx % print_every_steps == 0:
            return True

    if print_every_pct is not None:
        pct_now = 100.0 * step_idx / max(1, n_steps)
        pct_prev = 100.0 * (step_idx - 1) / max(1, n_steps)
        pct_bucket_now = int(pct_now / print_every_pct)
        pct_bucket_prev = int(pct_prev / print_every_pct)
        if pct_bucket_now > pct_bucket_prev:
            return True

    return False
