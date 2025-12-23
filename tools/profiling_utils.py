"""
HPC Profiling Utilities - Strict Measurement Discipline

Provides:
- Manifest generation (commit hash, environment, problem sizes)
- pstats export (cumtime, tottime tables)
- REPORT.md generation with hotspot categorization
- Automatic single-thread baseline verification

No speculation. No optimization without evidence.
"""

import os
import sys
import json
import subprocess
import platform
import pstats
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


def get_git_info() -> Dict[str, Any]:
    """Get git commit hash and dirty status. Fail if not a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        dirty_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        is_dirty = len(dirty_status) > 0

        return {
            "commit_hash": commit,
            "commit_short": commit[:8],
            "is_dirty": is_dirty,
            "dirty_files": dirty_status.split('\n') if is_dirty else []
        }
    except Exception as e:
        return {
            "commit_hash": "UNKNOWN",
            "commit_short": "UNKNOWN",
            "is_dirty": True,
            "error": str(e)
        }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU model and core count."""
    cpu_info = {"model": "UNKNOWN", "physical_cores": 0, "logical_cores": 0}

    # Try /proc/cpuinfo on Linux
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("model name"):
                        cpu_info["model"] = line.split(":")[1].strip()
                        break

                # Count physical cores
                cpu_info["physical_cores"] = subprocess.check_output(
                    ["grep", "-c", "^processor", "/proc/cpuinfo"],
                    text=True
                ).strip()
                cpu_info["physical_cores"] = int(cpu_info["physical_cores"])
        except:
            pass

    # Fallback to platform module
    if cpu_info["model"] == "UNKNOWN":
        cpu_info["model"] = platform.processor() or "UNKNOWN"

    try:
        import multiprocessing
        cpu_info["logical_cores"] = multiprocessing.cpu_count()
    except:
        pass

    return cpu_info


def get_thread_env_vars() -> Dict[str, str]:
    """Get thread-control environment variables (MUST be 1 for baseline)."""
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "NOT_SET"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "NOT_SET"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "NOT_SET"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "NOT_SET"),
    }


def verify_single_thread_baseline() -> Tuple[bool, List[str]]:
    """
    Verify that thread-control env vars are set to 1 for single-thread baseline.
    Returns (is_valid, warnings).
    """
    thread_env = get_thread_env_vars()
    warnings = []
    is_valid = True

    for key, val in thread_env.items():
        if val == "NOT_SET":
            warnings.append(f"WARNING: {key} is not set (should be 1 for baseline)")
            is_valid = False
        elif val != "1":
            warnings.append(f"WARNING: {key}={val} (should be 1 for baseline)")
            is_valid = False

    return is_valid, warnings


def create_manifest(
    output_dir: Path,
    command_line: str,
    problem_params: Dict[str, Any],
    wall_time_sec: float,
    cpu_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Create manifest.json with full provenance.

    Args:
        output_dir: Profiling output directory
        command_line: Full command line that was run
        problem_params: Dict with keys like ndof, nsteps, dt, n_elem, n_hinge, etc.
        wall_time_sec: Wall clock time in seconds
        cpu_time_sec: Optional CPU time in seconds
    """
    git_info = get_git_info()
    cpu_info = get_cpu_info()
    thread_env = get_thread_env_vars()
    is_baseline_valid, baseline_warnings = verify_single_thread_baseline()

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git": git_info,
        "command_line": command_line,
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": cpu_info,
        "thread_control_env": thread_env,
        "single_thread_baseline_valid": is_baseline_valid,
        "baseline_warnings": baseline_warnings,
        "packages": {},
        "problem_params": problem_params,
        "timing": {
            "wall_time_sec": wall_time_sec,
            "cpu_time_sec": cpu_time_sec,
        }
    }

    # Get package versions
    try:
        import numpy
        manifest["packages"]["numpy"] = numpy.__version__
    except:
        pass

    try:
        import scipy
        manifest["packages"]["scipy"] = scipy.__version__
    except:
        pass

    try:
        import numba
        manifest["packages"]["numba"] = numba.__version__
    except:
        pass

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[MANIFEST] Written to {manifest_path}")
    if not is_baseline_valid:
        print("[MANIFEST] ⚠️  SINGLE-THREAD BASELINE VALIDATION FAILED:")
        for w in baseline_warnings:
            print(f"           {w}")

    return manifest


def export_pstats(prof_file: Path, output_dir: Path, n_lines: int = 60):
    """
    Export pstats to human-readable text files.
    Creates:
        - pstats_cumtime.txt (top N by cumulative time)
        - pstats_tottime.txt (top N by total/self time)
    """
    stats = pstats.Stats(str(prof_file))

    # Cumulative time
    stream = io.StringIO()
    stats.stream = stream
    stats.sort_stats('cumulative')
    stats.print_stats(n_lines)
    cumtime_path = output_dir / "pstats_cumtime.txt"
    with open(cumtime_path, "w") as f:
        f.write(stream.getvalue())
    print(f"[PSTATS] Cumulative time table: {cumtime_path}")

    # Total/self time
    stream = io.StringIO()
    stats.stream = stream
    stats.sort_stats('tottime')
    stats.print_stats(n_lines)
    tottime_path = output_dir / "pstats_tottime.txt"
    with open(tottime_path, "w") as f:
        f.write(stream.getvalue())
    print(f"[PSTATS] Total time table: {tottime_path}")

    return stats


def parse_pstats_top_n(stats: pstats.Stats, sort_key: str, n: int = 20) -> List[Dict[str, Any]]:
    """
    Parse pstats to extract top N entries as structured data.

    Args:
        stats: pstats.Stats object
        sort_key: 'cumulative' or 'tottime'
        n: Number of entries to extract

    Returns:
        List of dicts with keys: func_name, ncalls, tottime, cumtime, percall_tot, percall_cum
    """
    stats.sort_stats(sort_key)

    # Extract stats
    results = []
    for func, (cc, nc, tt, ct, callers) in sorted(
        stats.stats.items(),
        key=lambda x: x[1][3 if sort_key == 'cumulative' else 2],
        reverse=True
    )[:n]:
        file_name, line_num, func_name = func
        results.append({
            "func_name": f"{file_name}:{line_num}({func_name})",
            "ncalls": f"{nc}/{cc}" if nc != cc else str(nc),
            "tottime": tt,
            "cumtime": ct,
            "percall_tot": tt / nc if nc > 0 else 0,
            "percall_cum": ct / nc if nc > 0 else 0,
        })

    return results


def categorize_hotspot(func_name: str, ncalls: int, cumtime: float, tottime: float) -> str:
    """
    Categorize a hotspot based on function name and call pattern.

    Categories:
    - Python loop/dispatch overhead
    - allocation pressure (repeated temporary arrays)
    - NumPy kernel dominated
    - linear algebra dominated
    - I/O dominated
    - postprocessing inside time loop
    """
    func_lower = func_name.lower()

    # Linear algebra
    if any(x in func_lower for x in ['solve', 'linalg', 'factorize', 'lu_', 'cholesky']):
        return "linear algebra dominated"

    # NumPy kernels
    if 'numpy' in func_lower or 'np.' in func_lower:
        if any(x in func_lower for x in ['dot', 'matmul', '@']):
            return "NumPy kernel dominated (matmul)"
        elif any(x in func_lower for x in ['array', 'zeros', 'empty', 'copy']):
            return "allocation pressure (repeated temporary arrays)"
        else:
            return "NumPy kernel dominated"

    # I/O
    if any(x in func_lower for x in ['write', 'open', 'save', 'dump', 'print']):
        return "I/O dominated"

    # Postprocessing
    if any(x in func_lower for x in ['plot', 'post', 'export', 'csv']):
        return "postprocessing inside time loop"

    # High call count suggests loop overhead
    if ncalls > 10000:
        return "Python loop/dispatch overhead"

    # Default
    return "application logic"


def generate_report(
    output_dir: Path,
    manifest: Dict[str, Any],
    stats: pstats.Stats,
    top_n: int = 20,
):
    """
    Generate REPORT.md with evidence tables and hotspot categorization.
    """
    report_path = output_dir / "REPORT.md"

    # Parse top entries
    top_cumtime = parse_pstats_top_n(stats, 'cumulative', top_n)
    top_tottime = parse_pstats_top_n(stats, 'tottime', top_n)

    with open(report_path, "w") as f:
        f.write("# HPC Profiling Report — Evidence-Based Analysis\n\n")
        f.write(f"**Generated**: {manifest['timestamp']}\n\n")
        f.write(f"**Commit**: `{manifest['git']['commit_short']}` ")
        if manifest['git']['is_dirty']:
            f.write("(⚠️  DIRTY)\n")
        else:
            f.write("(clean)\n")
        f.write(f"**Command**: `{manifest['command_line']}`\n\n")

        # Baseline validation
        f.write("## Baseline Validation\n\n")
        if manifest['single_thread_baseline_valid']:
            f.write("✅ Single-thread baseline validated.\n\n")
        else:
            f.write("❌ **SINGLE-THREAD BASELINE FAILED**\n\n")
            for w in manifest['baseline_warnings']:
                f.write(f"- {w}\n")
            f.write("\n")

        # Problem parameters
        f.write("## Problem Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for key, val in manifest['problem_params'].items():
            f.write(f"| {key} | {val} |\n")
        f.write("\n")

        # Timing summary
        f.write("## Timing Summary\n\n")
        wall_time = manifest['timing']['wall_time_sec']
        f.write(f"- **Wall time**: {wall_time:.3f} s\n")
        if manifest['timing']['cpu_time_sec'] is not None:
            f.write(f"- **CPU time**: {manifest['timing']['cpu_time_sec']:.3f} s\n")

        # Compute derived metrics if available
        if 'nsteps' in manifest['problem_params']:
            nsteps = manifest['problem_params']['nsteps']
            f.write(f"- **Steps/sec**: {nsteps / wall_time:.1f}\n")
        if 'dt' in manifest['problem_params'] and 'nsteps' in manifest['problem_params']:
            dt = manifest['problem_params']['dt']
            # Only compute simulation time if dt is numeric (not "static")
            if isinstance(dt, (int, float)):
                total_sim_time = dt * manifest['problem_params']['nsteps']
                f.write(f"- **Simulation time**: {total_sim_time:.3f} s\n")
        f.write("\n")

        # Top hotspots by cumulative time
        f.write("## Top Hotspots by Cumulative Time\n\n")
        f.write("| Rank | Function | ncalls | tottime | cumtime | percall (cum) |\n")
        f.write("|------|----------|--------|---------|---------|---------------|\n")
        for i, entry in enumerate(top_cumtime, 1):
            f.write(
                f"| {i} | `{entry['func_name']}` | {entry['ncalls']} | "
                f"{entry['tottime']:.4f} | {entry['cumtime']:.4f} | {entry['percall_cum']:.6f} |\n"
            )
        f.write("\n")

        # Top hotspots by self time
        f.write("## Top Hotspots by Self Time (tottime)\n\n")
        f.write("| Rank | Function | ncalls | tottime | cumtime | percall (tot) |\n")
        f.write("|------|----------|--------|---------|---------|---------------|\n")
        for i, entry in enumerate(top_tottime, 1):
            f.write(
                f"| {i} | `{entry['func_name']}` | {entry['ncalls']} | "
                f"{entry['tottime']:.4f} | {entry['cumtime']:.4f} | {entry['percall_tot']:.6f} |\n"
            )
        f.write("\n")

        # Detailed analysis of top 3 by cumulative time
        f.write("## Detailed Hotspot Analysis (Top 3 by Cumulative Time)\n\n")
        for i, entry in enumerate(top_cumtime[:3], 1):
            ncalls = int(entry['ncalls'].split('/')[0])
            category = categorize_hotspot(
                entry['func_name'], ncalls, entry['cumtime'], entry['tottime']
            )

            f.write(f"### {i}. `{entry['func_name']}`\n\n")
            f.write(f"- **Category**: {category}\n")
            f.write(f"- **ncalls**: {entry['ncalls']}\n")
            f.write(f"- **cumtime**: {entry['cumtime']:.4f} s ({100 * entry['cumtime'] / wall_time:.1f}% of total)\n")
            f.write(f"- **tottime**: {entry['tottime']:.4f} s (self)\n")
            f.write(f"- **Why hot**: Called {ncalls} times; likely inside timestep or Newton loop.\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("**Next steps**: Review detailed tables in `pstats_cumtime.txt` and `pstats_tottime.txt`.\n")

    print(f"[REPORT] Generated: {report_path}")
    return report_path
