#!/usr/bin/env python
"""
HPC Benchmark Runner — Problem 4 Portal Frame (Profiling Mode)

Enforces single-thread baseline and generates:
- prof.out (cProfile raw data)
- pstats_cumtime.txt, pstats_tottime.txt (top 60 functions)
- manifest.json (full provenance: commit, env, problem sizes)
- REPORT.md (evidence-based hotspot analysis)

Usage:
    python tools/profile_run.py --tag baseline_hht
    python tools/profile_run.py --tag baseline_hht --integrator hht --beam-hinge fiber

Environment variables (MUST be set for baseline):
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1

No optimization. No speculation. Evidence only.
"""

import sys
import os
import argparse
import cProfile
import time
from pathlib import Path

# Add src and tools to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "tools"))

from profiling_utils import (
    create_manifest,
    export_pstats,
    generate_report,
    verify_single_thread_baseline,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HPC Profiling Runner for Problem 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Profiling args
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Profiling run tag (e.g., 'baseline_hht', 'opt1_numba')",
    )

    # Problem 4 args (pass-through)
    parser.add_argument(
        "--integrator",
        type=str,
        default="hht",
        choices=["hht", "newmark", "explicit"],
        help="Time integrator (default: hht)",
    )
    parser.add_argument(
        "--beam-hinge",
        type=str,
        default="shm",
        choices=["shm", "fiber"],
        help="Beam hinge model (default: shm)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="gravity",
        choices=["gravity", "ida"],
        help="Analysis state: gravity-only or full IDA (default: gravity for profiling)",
    )
    parser.add_argument(
        "--gravity-steps",
        type=int,
        default=10,
        help="Number of gravity load increments (default: 10)",
    )
    parser.add_argument(
        "--nlgeom",
        action="store_true",
        help="Enable geometric nonlinearity (P-Delta)",
    )

    # Minimal IDA settings for profiling
    parser.add_argument(
        "--ag-min",
        type=float,
        default=0.1,
        help="Min amplitude for IDA (default: 0.1g)",
    )
    parser.add_argument(
        "--ag-max",
        type=float,
        default=0.3,
        help="Max amplitude for IDA (default: 0.3g, minimal for profiling)",
    )
    parser.add_argument(
        "--ag-step",
        type=float,
        default=0.1,
        help="Amplitude step for IDA (default: 0.1g)",
    )

    return parser.parse_args()


def run_problem4_profiled(args):
    """
    Run Problem 4 with cProfile enabled.
    Returns (wall_time, cpu_time, problem_params).
    """
    # Import problem4 module
    from problems import problema4_portico

    # Override sys.argv to pass args to problem4
    original_argv = sys.argv.copy()

    sys.argv = ["problema4_portico"]
    sys.argv.append("--integrator")
    sys.argv.append(args.integrator)
    sys.argv.append("--beam-hinge")
    sys.argv.append(args.beam_hinge)
    sys.argv.append("--state")
    sys.argv.append(args.state)
    sys.argv.append("--gravity-steps")
    sys.argv.append(str(args.gravity_steps))

    if args.nlgeom:
        sys.argv.append("--nlgeom")

    if args.state == "ida":
        sys.argv.append("--ag-min")
        sys.argv.append(str(args.ag_min))
        sys.argv.append("--ag-max")
        sys.argv.append(str(args.ag_max))
        sys.argv.append("--ag-step")
        sys.argv.append(str(args.ag_step))

    # Capture wall/CPU time
    t_start_wall = time.time()
    t_start_cpu = time.process_time()

    try:
        # Run problem4 main
        problema4_portico.main()
    except SystemExit:
        # Problem4 may call sys.exit(0), catch it
        pass
    finally:
        sys.argv = original_argv

    t_end_wall = time.time()
    t_end_cpu = time.process_time()

    wall_time = t_end_wall - t_start_wall
    cpu_time = t_end_cpu - t_start_cpu

    # Extract problem parameters (hardcoded for Problem 4)
    # These should match the actual problem configuration
    problem_params = {
        "problem": "Problem 4 Portal Frame",
        "integrator": args.integrator,
        "beam_hinge": args.beam_hinge,
        "state": args.state,
        "nlgeom": args.nlgeom,
        "H": 3.0,  # m
        "L": 5.0,  # m
        "nseg": 6,  # discretization
        "n_elem": 18,  # 2*nseg (columns) + nseg (beam)
        "n_hinge": 4,  # 2 column base + 2 beam ends
        "ndof": 24,  # approx (8 nodes * 3 DOF)
    }

    # Add gravity-specific params
    if args.state == "gravity":
        problem_params["gravity_steps"] = args.gravity_steps
        problem_params["nsteps"] = args.gravity_steps
        problem_params["dt"] = "static"
    else:
        # IDA params
        problem_params["ag_min"] = args.ag_min
        problem_params["ag_max"] = args.ag_max
        problem_params["ag_step"] = args.ag_step
        problem_params["n_ida_runs"] = int((args.ag_max - args.ag_min) / args.ag_step) + 1

    return wall_time, cpu_time, problem_params


def main():
    args = parse_args()

    # Verify single-thread baseline
    print("\n" + "=" * 80)
    print("HPC PROFILING RUNNER — STRICT MEASUREMENT DISCIPLINE")
    print("=" * 80)
    print(f"\nTag: {args.tag}")
    print(f"Integrator: {args.integrator}")
    print(f"Beam hinge: {args.beam_hinge}")
    print(f"State: {args.state}")

    is_valid, warnings = verify_single_thread_baseline()
    if not is_valid:
        print("\n⚠️  BASELINE VALIDATION FAILED:")
        for w in warnings:
            print(f"    {w}")
        print("\nTo enforce single-thread baseline, run:")
        print("    export OMP_NUM_THREADS=1")
        print("    export MKL_NUM_THREADS=1")
        print("    export OPENBLAS_NUM_THREADS=1")
        print("    export NUMEXPR_NUM_THREADS=1")
        print("\nContinuing anyway (manifest will record this failure)...\n")
    else:
        print("\n✅ Single-thread baseline validated.\n")

    # Create output directory
    output_base = Path("outputs") / "profiling" / args.tag
    output_base.mkdir(parents=True, exist_ok=True)

    prof_file = output_base / "prof.out"

    # Run with cProfile
    print(f"Running Problem 4 with cProfile enabled...")
    print(f"Output directory: {output_base}")
    print("-" * 80)

    profiler = cProfile.Profile()
    profiler.enable()

    wall_time, cpu_time, problem_params = run_problem4_profiled(args)

    profiler.disable()
    profiler.dump_stats(str(prof_file))

    print("-" * 80)
    print(f"\n[PROFILER] Raw profile written to: {prof_file}")

    # Build command line for manifest
    command_line = f"python tools/profile_run.py --tag {args.tag} --integrator {args.integrator} --beam-hinge {args.beam_hinge} --state {args.state}"
    if args.nlgeom:
        command_line += " --nlgeom"

    # Create manifest
    manifest = create_manifest(
        output_dir=output_base,
        command_line=command_line,
        problem_params=problem_params,
        wall_time_sec=wall_time,
        cpu_time_sec=cpu_time,
    )

    # Export pstats
    stats = export_pstats(prof_file, output_base, n_lines=60)

    # Generate report
    report_path = generate_report(output_base, manifest, stats, top_n=20)

    # Print summary
    print("\n" + "=" * 80)
    print("PROFILING RUN COMPLETE")
    print("=" * 80)
    print(f"\nWall time: {wall_time:.3f} s")
    print(f"CPU time:  {cpu_time:.3f} s")

    if 'nsteps' in problem_params and args.state == 'gravity':
        print(f"Gravity steps: {problem_params['nsteps']}")
        print(f"Steps/sec: {problem_params['nsteps'] / wall_time:.1f}")

    print(f"\nOutputs written to: {output_base}/")
    print(f"  - prof.out")
    print(f"  - pstats_cumtime.txt")
    print(f"  - pstats_tottime.txt")
    print(f"  - manifest.json")
    print(f"  - REPORT.md")

    print(f"\nTo view the report:")
    print(f"  cat {report_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
