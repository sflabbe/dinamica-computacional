#!/usr/bin/env python
"""Demo: JobRunner infrastructure with full Abaqus-style reporting.

This demonstrates:
- JobRunner context manager
- Output directory with timestamp
- File tracking (detects new files)
- Journal.log
- Abaqus-like .msg/.sta/.dat files
- Progress printing during analysis
- FLOPs estimation
- JSON-safe runinfo export (handles numpy arrays, etc.)

Run:
    python examples/demo_job_infrastructure.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import datetime

# Allow running as standalone script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dc_solver.job import JobRunner, print_progress, should_print_progress


def simulate_dynamics(
    n_steps: int = 5000,
    dt: float = 0.001,
    print_interval_pct: float = 5.0,
) -> dict:
    """Simulate a simple SDOF oscillator (demo only).

    Parameters
    ----------
    n_steps : int
        Number of time steps
    dt : float
        Time increment
    print_interval_pct : float
        Print progress every X percent

    Returns
    -------
    dict
        Results dictionary with t, u, v, a arrays
    """
    # Simple harmonic oscillator: m*a + c*v + k*u = 0
    # with m=1, c=0.1, k=100 -> omega_n ~ 10 rad/s, T ~ 0.6s
    m = 1.0
    c = 0.1
    k = 100.0

    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    t = np.zeros(n_steps)

    # Initial conditions
    u[0] = 1.0  # Initial displacement
    v[0] = 0.0
    a[0] = -(c * v[0] + k * u[0]) / m

    # Central difference time integration
    for i in range(1, n_steps):
        t[i] = i * dt

        # Explicit update
        u[i] = u[i-1] + dt * v[i-1] + 0.5 * dt**2 * a[i-1]
        f_ext = 0.0  # No external force
        f_int = k * u[i] + c * v[i-1]
        a[i] = (f_ext - f_int) / m
        v[i] = v[i-1] + 0.5 * dt * (a[i-1] + a[i])

        # Print progress periodically
        if should_print_progress(i, n_steps, print_every_pct=print_interval_pct):
            drift_peak = abs(u[:i+1].max())
            print_progress(
                step_idx=i,
                n_steps=n_steps,
                t=t[i],
                dt=dt,
                substep=0,
                drift_peak=drift_peak * 100.0,
                extra={"u_max": u[i], "v_max": v[i]},
            )

    return {"t": t, "u": u, "v": v, "a": a}


def main():
    """Main demo function."""
    # Create unique output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / "demo_job" / f"{timestamp}__explicit__sdof"

    # Metadata to include in run info
    meta = {
        "problem": "demo_sdof_oscillator",
        "integrator": "central_difference",
        "m": 1.0,
        "c": 0.1,
        "k": 100.0,
        "initial_u": 1.0,
        # Include numpy array to test JSON-safe serialization
        "test_array": np.array([1.0, 2.0, 3.0]),
    }

    # Use JobRunner context manager
    with JobRunner(job_name="demo_job", output_dir=output_dir, meta=meta) as job:
        # Set analysis parameters for FLOPs estimation
        n_steps = 5000
        job.set_analysis_params(
            ndof=1,  # SDOF system
            n_steps=n_steps,
            integrator="explicit",
            avg_iterations=1.0,
        )

        job.log("Starting SDOF simulation...")

        # Run simulation
        results = simulate_dynamics(n_steps=n_steps, dt=0.001, print_interval_pct=5.0)

        job.log(f"Simulation complete. Generated {len(results['t'])} time points.")

        # Plot results
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        axes[0].plot(results["t"], results["u"])
        axes[0].set_ylabel("Displacement u")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(results["t"], results["v"])
        axes[1].set_ylabel("Velocity v")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(results["t"], results["a"])
        axes[2].set_ylabel("Acceleration a")
        axes[2].set_xlabel("Time [s]")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("SDOF Oscillator Response", y=1.0)
        fig.tight_layout()

        plot_path = output_dir / "demo_sdof_response.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        job.log(f"Saved plot: {plot_path.name}")

        # Export CSV
        csv_path = output_dir / "demo_sdof_timeseries.csv"
        with csv_path.open("w") as f:
            f.write("t,u,v,a\n")
            for i in range(len(results["t"])):
                f.write(f"{results['t'][i]:.6f},{results['u'][i]:.6e},{results['v'][i]:.6e},{results['a'][i]:.6e}\n")

        job.log(f"Saved CSV: {csv_path.name}")

        # Mark job as successful
        job.mark_success()

    print("\n" + "=" * 88)
    print("DEMO COMPLETE!")
    print(f"Check outputs in: {output_dir}")
    print("Files generated:")
    print(f"  - demo_job.msg      (messages and iteration details)")
    print(f"  - demo_job.sta      (status file, incremental)")
    print(f"  - demo_job.dat      (data file with JOB TOTALS)")
    print(f"  - journal.log       (chronological event log)")
    print(f"  - demo_job_runinfo.json/.txt  (run metadata, JSON-safe)")
    print(f"  - demo_sdof_response.png")
    print(f"  - demo_sdof_timeseries.csv")
    print("=" * 88)


if __name__ == "__main__":
    main()
