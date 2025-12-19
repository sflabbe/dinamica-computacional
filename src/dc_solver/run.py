"""CLI entry point for Abaqus-like input runs."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np

from dc_solver.io.abaqus_inp import parse_inp, build_model, apply_gravity, apply_cloads, amplitude_series
from dc_solver.integrators.hht_alpha import hht_alpha_newton
from dc_solver.post.plotting import plot_structure_states, write_member_stress_csv
from dc_solver.reporting import (
    AbaqusLikeReporter,
    Error,
    IncrementEnd,
    IncrementStart,
    InputEcho,
    JobEnd,
    JobStart,
    StepEnd,
    StepStart,
    Warning,
)
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


def _detect_beam_benchmark(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    if "beam_cantilever" in name:
        return "cantilever"
    if "beam_simply_supported" in name:
        return "simply_supported"
    return None


def beam_benchmark_report(
    model,
    data,
    u: np.ndarray,
    benchmark_kind: str,
) -> Dict[str, float]:
    xs = np.array([nd.x for nd in model.nodes], dtype=float)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    L = max_x - min_x
    E = float(data.material.E)
    I = float(model.beams[0].I) if model.beams else 0.0

    report: Dict[str, float] = {"L": L, "E": E, "I": I}
    if benchmark_kind == "cantilever":
        tip_idx = int(np.argmax(xs))
        tip_node = model.nodes[tip_idx]
        uy_tip = float(u[tip_node.dof_u[1]])
        th_tip = float(u[tip_node.dof_th])
        P = float(model.load_const[tip_node.dof_u[1]])
        delta = P * L**3 / (3.0 * E * I)
        theta = P * L**2 / (2.0 * E * I)
        report.update({
            "uy_tip": uy_tip,
            "theta_tip": th_tip,
            "P": P,
            "theory_uy_tip": delta,
            "theory_theta_tip": theta,
        })
    elif benchmark_kind == "simply_supported":
        mid_x = min_x + 0.5 * L
        mid_idx = int(np.argmin(np.abs(xs - mid_x)))
        mid_node = model.nodes[mid_idx]
        uy_mid = float(u[mid_node.dof_u[1]])
        P = float(model.load_const[mid_node.dof_u[1]])
        delta = P * L**3 / (48.0 * E * I)
        report.update({
            "uy_mid": uy_mid,
            "P": P,
            "theory_uy_mid": delta,
        })
    return report


def _print_beam_report(benchmark_kind: str, report: Dict[str, float]) -> None:
    if benchmark_kind == "cantilever":
        uy_tip = report["uy_tip"]
        th_tip = report["theta_tip"]
        delta = report["theory_uy_tip"]
        theta = report["theory_theta_tip"]
        err_uy = 100.0 * (uy_tip - delta) / delta if delta != 0 else float("nan")
        err_th = 100.0 * (th_tip - theta) / theta if theta != 0 else float("nan")
        print("Beam benchmark: cantilever tip load")
        print(f"  uy_tip [m] = {uy_tip:.6e} (theory {delta:.6e}, err {err_uy:+.3f}%)")
        print(f"  theta_tip [rad] = {th_tip:.6e} (theory {theta:.6e}, err {err_th:+.3f}%)")
    elif benchmark_kind == "simply_supported":
        uy_mid = report["uy_mid"]
        delta = report["theory_uy_mid"]
        err_uy = 100.0 * (uy_mid - delta) / delta if delta != 0 else float("nan")
        print("Beam benchmark: simply supported midspan load")
        print(f"  uy_mid [m] = {uy_mid:.6e} (theory {delta:.6e}, err {err_uy:+.3f}%)")


def _input_echo_lines(data) -> List[str]:
    lines = [f"*PART, NAME={data.part.name}"]
    lines.append(f"*NODE, NSET=ALLNODES ({len(data.part.nodes)} total)")
    lines.append(f"*ELEMENT, TYPE=B31 ({len(data.part.elements)} total)")
    for elset, section in data.part.beam_section.items():
        section_name, b, h, mat = section
        lines.append(
            f"*BEAM SECTION, ELSET={elset}, MATERIAL={mat}, SECTION={section_name}"
        )
        lines.append(f"{b}, {h}")
    lines.append(f"*MATERIAL, NAME={data.material.name}")
    lines.append("*ELASTIC")
    lines.append(f"{data.material.E}, {data.material.nu}")
    if data.material.density:
        lines.append("*DENSITY")
        lines.append(f"{data.material.density}")
    lines.append("*ASSEMBLY")
    tx, ty = data.assembly_translation
    lines.append("*INSTANCE")
    lines.append(f"{tx}, {ty}")
    for name, dof1, dof2, value in data.boundaries:
        lines.append("*BOUNDARY")
        lines.append(f"{name}, {dof1}, {dof2}, {value}")
    for step in data.steps:
        lines.append(f"*STEP, NAME={step.name}, NLGEOM={'YES' if step.nlgeom else 'NO'}")
        if step.kind == "STATIC":
            lines.append("*STATIC")
        else:
            lines.append("*DYNAMIC")
            lines.append(f"{step.dt}, {step.time_period}")
        if step.gravity is not None:
            gx, gy = step.gravity
            lines.append("*DLOAD, GRAV")
            lines.append(f"{gx}, {gy}")
        if step.accel_bc is not None:
            set_name, dof, value, amp_name = step.accel_bc
            amp_label = f", AMPLITUDE={amp_name}" if amp_name else ""
            lines.append(f"*BOUNDARY, TYPE=ACCELERATION{amp_label}")
            lines.append(f"{set_name}, {dof}, {dof}, {value}")
        if step.cloads:
            lines.append("*CLOAD")
            for target, dof, value in step.cloads:
                lines.append(f"{target}, {dof}, {value}")
        lines.append("*END STEP")
    return lines


def run_inp(path: str, abaqus_like_logs: bool = False, output_dir: Optional[str] = None) -> None:
    reporter = None
    exc: Optional[Exception] = None
    job_name = Path(path).stem
    basepath = Path(output_dir) if output_dir else Path.cwd()
    start_dt = datetime.now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if abaqus_like_logs:
        reporter = AbaqusLikeReporter(basepath, job_name, clock=datetime.now)
        reporter.on_event(
            JobStart(
                job_name=job_name,
                start_dt=start_dt,
                solver_version="dev",
                cwd=str(Path.cwd()),
                output_dir=str(basepath),
            )
        )

    def warning_cb(msg: str) -> None:
        if reporter is not None:
            reporter.on_event(Warning(msg, phase="INPUT"))
        else:
            print(msg)

    try:
        data = parse_inp(path, warning_cb=warning_cb if reporter is not None else None)
        if reporter is not None:
            reporter.on_event(InputEcho(_input_echo_lines(data)))

        last = None
        u_static = None

        current_gravity = None
        benchmark_kind = _detect_beam_benchmark(path)
        benchmark_report = None
        total_time = 0.0
        for step_id, step in enumerate(data.steps, start=1):
            model = build_model(data, nlgeom=step.nlgeom)
            step_time = 0.0

            if step.gravity is not None:
                current_gravity = step.gravity
            if current_gravity is not None:
                apply_gravity(model, data, current_gravity)

            if reporter is not None:
                max_increments = 1
                if step.kind == "DYNAMIC" and step.dt > 0:
                    max_increments = int(round(step.time_period / step.dt))
                reporter.on_event(
                    StepStart(
                        step_id=step_id,
                        step_name=step.name,
                        procedure="STATIC" if step.kind == "STATIC" else "DYNAMIC IMPLICIT",
                        total_time=total_time,
                        nlgeom=step.nlgeom,
                        dt0=step.dt if step.kind == "DYNAMIC" else 1.0,
                        dtmin=step.dt if step.kind == "DYNAMIC" else 1.0,
                        dtmax=step.dt if step.kind == "DYNAMIC" else 1.0,
                        max_increments=max_increments,
                    )
                )

            if step.kind == "STATIC":
                if step.cloads:
                    apply_cloads(model, data, step)
                if reporter is not None:
                    reporter.on_event(
                        IncrementStart(
                            step_id=step_id,
                            inc=1,
                            attempt=1,
                            dt=1.0,
                            step_time=0.0,
                            total_time=total_time,
                            is_cutback_attempt=False,
                        )
                    )
                stats = {}
                u_static = solve_static_newton(
                    model,
                    model.load_const,
                    reporter=reporter.on_event if reporter is not None else None,
                    step_id=step_id,
                    inc=1,
                    attempt=1,
                    stats=stats,
                )
                last = {"t": np.array([0.0]), "u": np.array([u_static]), "drift": np.array([0.0])}
                step_time = 1.0
                total_time += step_time
                if reporter is not None:
                    reporter.on_event(
                        IncrementEnd(
                            step_id=step_id,
                            inc=1,
                            attempt=1,
                            converged=True,
                            n_equil_iters=int(stats.get("iters", 1)),
                            n_severe_iters=0,
                            dt_completed=1.0,
                            step_fraction=1.0,
                            step_time_completed=1.0,
                            total_time_completed=total_time,
                        )
                    )
                if benchmark_kind:
                    benchmark_report = beam_benchmark_report(model, data, u_static, benchmark_kind)
                    _print_beam_report(benchmark_kind, benchmark_report)
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
                    reporter=reporter.on_event if reporter is not None else None,
                    step_id=step_id,
                )
                step_time = float(t[-1]) if t.size else 0.0
                total_time += step_time
            else:
                raise ValueError(f"Unsupported step kind {step.kind}")

            if reporter is not None:
                reporter.on_event(
                    StepEnd(
                        step_id=step_id,
                        step_time_completed=step_time,
                        total_time_completed=total_time,
                    )
                )

        if last is not None:
            base_nodes, drift_nodes, height = _extreme_nodes_by_y(model)
            prefix = Path(path).with_suffix("")
            plot_structure_states(
                model,
                last,
                drift_height=height,
                snapshot_limit=0.04,
                outfile=f"{prefix.name}_states.png",
                benchmark_kind=benchmark_kind,
                benchmark_report=benchmark_report,
                field="both",
            )
            plot_structure_states(
                model,
                last,
                drift_height=height,
                snapshot_limit=0.04,
                outfile=f"{prefix.name}_states_U.png",
                benchmark_kind=benchmark_kind,
                benchmark_report=benchmark_report,
                field="U",
                shared_colorbar=True,
            )
            plot_structure_states(
                model,
                last,
                drift_height=height,
                snapshot_limit=0.04,
                outfile=f"{prefix.name}_states_S.png",
                benchmark_kind=benchmark_kind,
                benchmark_report=benchmark_report,
                field="S",
                shared_colorbar=True,
            )
            u_last = last["u"][-1]
            write_member_stress_csv(model, u_last, f"{prefix.name}_member_stress.csv")
    except Exception as err:
        exc = err
        if reporter is not None:
            reporter.on_event(Error(str(err)))
    finally:
        if reporter is not None:
            end_dt = datetime.now()
            wall_s = time.perf_counter() - wall_start
            cpu_user_s = time.process_time() - cpu_start
            reporter.on_event(
                JobEnd(
                    success=exc is None,
                    end_dt=end_dt,
                    cpu_user_s=cpu_user_s,
                    cpu_sys_s=0.0,
                    wall_s=wall_s,
                    warnings_count=reporter.warnings_count,
                    errors_count=reporter.errors_count,
                    totals={},
                )
            )
            reporter.close()
    if exc is not None:
        raise exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Abaqus-like .inp model.")
    parser.add_argument("inp_path", help="Path to .inp file")
    parser.add_argument(
        "--abaqus-like-logs",
        action="store_true",
        help="Generate Abaqus-like .sta/.msg/.dat logs in the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output logs when using --abaqus-like-logs.",
    )
    args = parser.parse_args()
    run_inp(args.inp_path, abaqus_like_logs=args.abaqus_like_logs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
