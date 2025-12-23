#!/usr/bin/env python3
"""Benchmark runner with cProfile integration for problema4_portico.

Usage:
    # IDA run (2 levels: 0.1g, 0.2g)
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    python tools/profile_run.py --tag baseline --state ida --ag-min 0.1 --ag-max 0.2 --ag-step 0.1

    # Gravity-only run (quick test)
    python tools/profile_run.py --tag gravity_test --state gravity --gravity-steps 10

Outputs:
    outputs/profiling/<tag>/
        prof.out              # cProfile binary
        pstats_cumtime.txt    # top 60 by cumulative time
        pstats_tottime.txt    # top 60 by total/self time
        manifest.json         # run metadata
        REPORT.md             # hotspot analysis
"""

from __future__ import annotations

import argparse
import cProfile
import os
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from profiling_utils import create_manifest, export_pstats_top, create_report


def run_problema4_with_profiling(args: argparse.Namespace) -> tuple[float, dict]:
    """Run problema4_portico.py with cProfile and return wall time + problem sizes."""
    # Build command line for problema4
    problema4_args = [
        "--state", args.state,
        "--integrator", args.integrator,
        "--beam-hinge", args.beam_hinge,
    ]

    if args.state == "ida":
        problema4_args.extend([
            "--ag-min", str(args.ag_min),
            "--ag-max", str(args.ag_max),
            "--ag-step", str(args.ag_step),
        ])
    elif args.state == "gravity":
        problema4_args.extend([
            "--gravity-steps", str(args.gravity_steps),
        ])

    # Add optional args
    if args.t_end is not None:
        problema4_args.extend(["--t-end", str(args.t_end)])
    if args.base_dt is not None:
        problema4_args.extend(["--base-dt", str(args.base_dt)])

    # Run with profiling
    profiler = cProfile.Profile()
    print(f"Starting profiled run: {args.tag}")
    print(f"Command: python -m problems.problema4_portico {' '.join(problema4_args)}")
    print(f"Single-thread enforcement: checking env vars...")

    # Check threading environment
    thread_vars = {
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS"),
    }
    print("Thread control:")
    for k, v in thread_vars.items():
        status = "✓" if v == "1" else "⚠"
        print(f"  {status} {k}={v or 'unset'}")

    if not all(v == "1" for v in thread_vars.values()):
        print("\n⚠ WARNING: Not all thread control vars are set to 1. Baseline may not be single-threaded.")

    print("\nStarting profiled execution...\n")

    # Temporarily modify sys.argv to pass args to problema4
    old_argv = sys.argv
    sys.argv = ["problema4_portico.py"] + problema4_args

    t0 = time.time()
    profiler.enable()

    try:
        # Import and run problema4 main
        from problems.problema4_portico import main as problema4_main
        problema4_main()
    finally:
        profiler.disable()
        sys.argv = old_argv

    wall_time = time.time() - t0

    print(f"\n✓ Profiled run completed in {wall_time:.2f} s")

    # Extract problem sizes from args
    problem_sizes = {
        "state": args.state,
        "integrator": args.integrator,
        "beam_hinge": args.beam_hinge,
    }

    if args.state == "ida":
        n_runs = int(round((args.ag_max - args.ag_min) / args.ag_step)) + 1
        problem_sizes["ida_runs"] = n_runs
        problem_sizes["ag_range"] = f"{args.ag_min}-{args.ag_max}g (step {args.ag_step}g)"
    elif args.state == "gravity":
        problem_sizes["gravity_steps"] = args.gravity_steps

    return wall_time, problem_sizes, profiler


def main():
    parser = argparse.ArgumentParser(
        description="Profile problema4_portico.py with cProfile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Profiling args
    parser.add_argument("--tag", required=True, help="Tag for this profiling run (output dir name)")

    # Problema4 args (subset)
    parser.add_argument("--state", default="ida", choices=["ida", "gravity"],
                        help="Run mode: 'ida' (IDA) or 'gravity' (gravity-only)")
    parser.add_argument("--integrator", default="hht", choices=["hht", "newmark", "explicit"],
                        help="Time integrator")
    parser.add_argument("--beam-hinge", default="shm", choices=["shm", "fiber", "compare"],
                        help="Beam hinge model")

    # IDA args
    parser.add_argument("--ag-min", type=float, default=0.1, help="Minimum Ag [g]")
    parser.add_argument("--ag-max", type=float, default=0.2, help="Maximum Ag [g]")
    parser.add_argument("--ag-step", type=float, default=0.1, help="Ag step [g]")

    # Gravity args
    parser.add_argument("--gravity-steps", type=int, default=10, help="Gravity load steps")

    # Optional dynamic args
    parser.add_argument("--t-end", type=float, help="End time [s]")
    parser.add_argument("--base-dt", type=float, help="Base time step [s]")

    args = parser.parse_args()

    # Create output directory
    output_dir = REPO_ROOT / "outputs" / "profiling" / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Run profiled execution
    wall_time, problem_sizes, profiler = run_problema4_with_profiling(args)

    # Save cProfile output
    prof_out = output_dir / "prof.out"
    profiler.dump_stats(str(prof_out))
    print(f"✓ cProfile output saved to {prof_out}")

    # Export pstats top N
    export_pstats_top(prof_out, output_dir, top_n=60)

    # Create manifest
    command_line = " ".join(sys.argv)
    create_manifest(
        tag=args.tag,
        command_line=command_line,
        wall_time=wall_time,
        problem_sizes=problem_sizes,
        output_dir=output_dir,
    )

    # Create report
    manifest_file = output_dir / "manifest.json"
    create_report(
        tag=args.tag,
        stats_file=prof_out,
        manifest_file=manifest_file,
        output_dir=output_dir,
    )

    print(f"\n{'='*70}")
    print(f"✓ Profiling complete: {args.tag}")
    print(f"  Wall time: {wall_time:.2f} s")
    print(f"  Outputs: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
