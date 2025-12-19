"""Run the portal frame problem using the new engine + Abaqus-like input."""

from __future__ import annotations

import argparse

import numpy as np

from dinamica_computacional.core.analysis import run_analysis
from dinamica_computacional.io.abaqus_like import read_inp


def _summarize_dynamic(step_name: str, data: dict) -> str:
    t = data.get("t", np.array([], dtype=float))
    drift = data.get("drift", np.array([], dtype=float))
    vb = data.get("Vb", np.array([], dtype=float))
    snapshot_t = float(data.get("snapshot_t", float("nan")))
    snapshot_drift = float(data.get("snapshot_drift", float("nan")))

    if drift.size:
        peak_idx = int(np.argmax(np.abs(drift)))
        peak_drift = float(drift[peak_idx])
        peak_t = float(t[peak_idx]) if t.size else float("nan")
    else:
        peak_drift = float("nan")
        peak_t = float("nan")

    peak_vb = float(np.max(np.abs(vb))) if vb.size else float("nan")

    return (
        f"Dynamic step: {step_name}\n"
        f"  Peak drift = {100.0 * peak_drift:.2f}% at t={peak_t:.3f}s\n"
        f"  Peak Vb = {peak_vb / 1e3:.2f} kN\n"
        f"  Snapshot drift = {100.0 * snapshot_drift:.2f}% at t={snapshot_t:.3f}s\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Portal frame demo using new engine + input format.")
    parser.add_argument(
        "--input",
        default="inputs/portal_problem4.inp",
        help="Path to Abaqus-like input file.",
    )
    parser.add_argument(
        "--prefix",
        default="problem4",
        help="Output prefix for generated plots.",
    )
    args = parser.parse_args()

    model, plan = read_inp(args.input)
    results = run_analysis(model, plan)
    results.save_plots(outfile_prefix=args.prefix)

    if results.dynamic_steps:
        last_step = list(results.dynamic_steps.keys())[-1]
        print(_summarize_dynamic(last_step, results.dynamic_steps[last_step]))


if __name__ == "__main__":
    main()
