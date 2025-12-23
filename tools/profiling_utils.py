#!/usr/bin/env python3
"""Profiling utilities: manifest generation, report creation, environment capture."""

from __future__ import annotations

import json
import platform
import pstats
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def get_git_info() -> Dict[str, str]:
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit_short = commit[:7]

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status)

        return {
            "commit": commit,
            "commit_short": commit_short,
            "dirty": dirty,
            "status": "dirty" if dirty else "clean",
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": "unknown",
            "commit_short": "unknown",
            "dirty": False,
            "status": "no-git",
        }


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    cpu_info = {"model": "unknown", "cores": -1}

    # Try to get CPU model from /proc/cpuinfo (Linux)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_info["model"] = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass

    # Fallback: use platform info
    if cpu_info["model"] == "unknown":
        cpu_info["model"] = platform.processor() or "unknown"

    # Get core count
    import os
    cpu_info["cores"] = os.cpu_count() or -1

    return cpu_info


def get_env_vars() -> Dict[str, str]:
    """Get relevant environment variables for threading control."""
    import os
    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DC_FAST",
    ]
    return {k: os.getenv(k, "unset") for k in env_keys}


def create_manifest(
    tag: str,
    command_line: str,
    wall_time: float,
    problem_sizes: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create manifest.json with run metadata."""
    import numpy as np

    git_info = get_git_info()
    cpu_info = get_cpu_info()
    env_vars = get_env_vars()

    manifest = {
        "tag": tag,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git": git_info,
        "command": command_line,
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "os": platform.system(),
            "cpu": cpu_info,
            "env_vars": env_vars,
        },
        "problem_sizes": problem_sizes,
        "timing": {
            "wall_time_seconds": wall_time,
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Manifest written to {manifest_path}")


def export_pstats_top(stats_file: Path, output_dir: Path, top_n: int = 60) -> None:
    """Export top N functions by cumtime and tottime to text files."""
    p = pstats.Stats(str(stats_file))

    # Top by cumulative time
    cumtime_path = output_dir / "pstats_cumtime.txt"
    with open(cumtime_path, "w") as f:
        sys.stdout = f
        p.strip_dirs().sort_stats("cumulative").print_stats(top_n)
        sys.stdout = sys.__stdout__
    print(f"✓ Top {top_n} by cumtime written to {cumtime_path}")

    # Top by total/self time
    tottime_path = output_dir / "pstats_tottime.txt"
    with open(tottime_path, "w") as f:
        sys.stdout = f
        p.strip_dirs().sort_stats("tottime").print_stats(top_n)
        sys.stdout = sys.__stdout__
    print(f"✓ Top {top_n} by tottime written to {tottime_path}")


def generate_hotspot_table(stats_file: Path, sort_by: str = "cumulative", top_n: int = 20) -> str:
    """Generate markdown table of top N hotspots."""
    p = pstats.Stats(str(stats_file))
    p.strip_dirs().sort_stats(sort_by)

    lines = [
        f"## Top {top_n} Functions (sorted by {sort_by})",
        "",
        "| Rank | Function | tottime (s) | cumtime (s) | ncalls | tottime/call (ms) |",
        "|------|----------|-------------|-------------|--------|-------------------|",
    ]

    stats = p.stats
    sorted_stats = sorted(stats.items(), key=lambda x: x[1][3], reverse=True) if sort_by == "cumulative" else sorted(stats.items(), key=lambda x: x[1][2], reverse=True)

    for i, (func, (cc, nc, tt, ct, callers)) in enumerate(sorted_stats[:top_n], 1):
        file, line, func_name = func
        file = Path(file).name  # basename only
        tt_per_call = (tt / nc * 1000) if nc > 0 else 0
        lines.append(
            f"| {i} | `{func_name}` ({file}:{line}) | {tt:.2f} | {ct:.2f} | {nc} | {tt_per_call:.3f} |"
        )

    return "\n".join(lines)


def create_report(
    tag: str,
    stats_file: Path,
    manifest_file: Path,
    output_dir: Path,
) -> None:
    """Create REPORT.md with hotspot analysis."""
    with open(manifest_file) as f:
        manifest = json.load(f)

    git_info = manifest["git"]
    env_vars = manifest["environment"]["env_vars"]
    timing = manifest["timing"]
    problem_sizes = manifest.get("problem_sizes", {})

    lines = [
        f"# Profiling Report: {tag}",
        "",
        f"**Date**: {manifest['timestamp']}",
        f"**Git**: {git_info['commit_short']} ({git_info['status']})",
        f"**Command**: `{manifest['command']}`",
        "",
        "## Environment",
        "",
        f"- Python: {manifest['environment']['python_version'].split()[0]}",
        f"- NumPy: {manifest['environment']['numpy_version']}",
        f"- Platform: {manifest['environment']['platform']}",
        f"- CPU: {manifest['environment']['cpu']['model']} ({manifest['environment']['cpu']['cores']} cores)",
        "",
        "### Threading Control",
        "",
    ]

    for k, v in env_vars.items():
        lines.append(f"- `{k}={v}`")

    lines.extend([
        "",
        "## Problem Sizes",
        "",
    ])

    if problem_sizes:
        for k, v in problem_sizes.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("(not available)")

    lines.extend([
        "",
        "## Timing",
        "",
        f"**Wall time**: {timing['wall_time_seconds']:.2f} s",
        "",
        "---",
        "",
    ])

    # Add hotspot tables
    lines.append(generate_hotspot_table(stats_file, sort_by="cumulative", top_n=20))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(generate_hotspot_table(stats_file, sort_by="tottime", top_n=20))

    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✓ Report written to {report_path}")
