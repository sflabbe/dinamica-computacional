"""CLI entry point for Abaqus-like input runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from dc_solver.io.abaqus_inp import parse_inp, build_model, apply_gravity, apply_cloads, amplitude_series
from dc_solver.integrators.hht_alpha import hht_alpha_newton
from dc_solver.post.plotting import plot_structure_states, write_member_stress_csv
from dc_solver.static.newton import solve_static_newton


def _extreme_nodes_by_y(model) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
    ys = np.array([nd.y for nd in model.nodes])
    min_y = np.min(ys)
    max_y = np.max(ys)
    base_nodes = np.where(np.isclose(ys, min_y))[0].tolist()
    top_nodes = np.where(np.isclose(ys, max_y))[0].tolist()
    if len(base_nodes) < 2:
        base_nodes = [int(np.argmin(ys)), int(np.argmax(ys))]
    if len(top_nodes) < 2:
        top_nodes = [int(np.argmax(ys)), int(np.argmin(ys))]
    return (base_nodes[0], base_nodes[1]), (top_nodes[0], top_nodes[1]), float(max_y - min_y)


def run_inp(path: str) -> None:
    data = parse_inp(path)

    last = None
    u_static = None

    current_gravity = None
    for step in data.steps:
        model = build_model(data, nlgeom=step.nlgeom)

        if step.gravity is not None:
            current_gravity = step.gravity
        if current_gravity is not None:
            apply_gravity(model, data, current_gravity)

        if step.kind == "STATIC":
            if step.cloads:
                apply_cloads(model, data, step)
            u_static = solve_static_newton(model, model.load_const)
            last = {"t": np.array([0.0]), "u": np.array([u_static]), "drift": np.array([0.0])}
        elif step.kind == "DYNAMIC":
            if step.dt <= 0 or step.time_period <= 0:
                raise ValueError("Dynamic step requires time period and dt.")
            if step.accel_bc is None:
                raise ValueError("Dynamic step requires *Boundary,type=ACCELERATION.")
            _, dof, scale, amp_name = step.accel_bc
            amplitude = data.amplitudes.get(amp_name or "", [])
            t = np.arange(0.0, step.time_period + 1e-12, step.dt)
            ag = amplitude_series(amplitude, step.dt, step.time_period) * scale
            if dof != 1:
                raise ValueError("Only DOF=1 (ux) acceleration supported in this runner.")

            base_nodes, drift_nodes, height = _extreme_nodes_by_y(model)
            last = hht_alpha_newton(
                model,
                t,
                ag,
                drift_height=height,
                base_nodes=base_nodes,
                drift_nodes=drift_nodes,
                drift_limit=0.10,
                drift_snapshot=0.04,
                alpha=-0.05,
                max_iter=40,
                tol=1e-6,
                verbose=False,
            )
        else:
            raise ValueError(f"Unsupported step kind {step.kind}")

    if last is not None:
        base_nodes, drift_nodes, height = _extreme_nodes_by_y(model)
        prefix = Path(path).with_suffix("")
        plot_structure_states(
            model,
            last,
            drift_height=height,
            snapshot_limit=0.04,
            outfile=f"{prefix.name}_states.png",
        )
        u_last = last["u"][-1]
        write_member_stress_csv(model, u_last, f"{prefix.name}_member_stress.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Abaqus-like .inp model.")
    parser.add_argument("inp_path", help="Path to .inp file")
    args = parser.parse_args()
    run_inp(args.inp_path)


if __name__ == "__main__":
    main()
