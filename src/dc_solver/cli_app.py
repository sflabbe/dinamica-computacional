"""Interactive CLI for running and inspecting simulations."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dc_solver.io.abaqus_inp import (
    parse_inp,
    build_model,
    apply_gravity,
    apply_cloads,
    amplitude_series,
)
from dc_solver.integrators.hht_alpha import hht_alpha_newton
from dc_solver.post.plotting import (
    plot_model_assembly,
    plot_structure_state,
    plot_hinge_hysteresis,
    plot_hinge_nm_interaction,
)
from dc_solver.static.newton import solve_static_newton


def _outputs_dir(base: Optional[str] = None) -> Path:
    root = Path(base) if base else Path.cwd()
    out = root / "outputs" / "cli"
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class SimulationSession:
    output_dir: Path
    verbose: bool = False
    data: Optional[object] = None
    model: Optional[object] = None
    last: Optional[Dict[str, np.ndarray]] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def load(self, path: str) -> None:
        self.data = parse_inp(path)
        self.last = None
        self.model = None
        self.meta = {"path": str(path)}

    def run(self) -> None:
        if self.data is None:
            raise RuntimeError("No input loaded. Use 'open <file>' first.")

        data = self.data
        last = None
        model = None
        total_time = 0.0
        current_gravity = None

        for step in data.steps:
            model = build_model(data, nlgeom=step.nlgeom)
            step_time = 0.0

            if step.gravity is not None:
                current_gravity = step.gravity
            if current_gravity is not None:
                apply_gravity(model, data, current_gravity)

            if step.kind == "STATIC":
                if step.cloads:
                    apply_cloads(model, data, step)
                stats: Dict[str, float] = {}
                u_static = solve_static_newton(
                    model,
                    model.load_const,
                    reporter=None,
                    step_id=1,
                    inc=1,
                    attempt=1,
                    stats=stats,
                )
                last = {
                    "t": np.array([0.0]),
                    "u": np.array([u_static]),
                    "drift": np.array([0.0]),
                }
                step_time = 1.0
                total_time += step_time
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
                    verbose=self.verbose,
                )
                step_time = float(t[-1]) if t.size else 0.0
                total_time += step_time
            else:
                raise ValueError(f"Unsupported step kind {step.kind}")

        if last is None or model is None:
            raise RuntimeError("No results generated from the input.")

        self.last = last
        self.model = model
        self.meta["total_time"] = total_time

    def plot_assembly(self) -> Path:
        if self.data is None:
            raise RuntimeError("No input loaded.")
        if self.model is None:
            self.model = build_model(self.data, nlgeom=False)
        out = self.output_dir / "assembly.png"
        plot_model_assembly(self.model, outfile=str(out))
        return out

    def plot_hinges(self) -> List[Path]:
        if self.model is None or self.last is None:
            raise RuntimeError("Run a simulation before plotting hinge results.")
        outputs = []
        out_hyst = self.output_dir / "hinge_hysteresis.png"
        plot_hinge_hysteresis(self.model, self.last, outfile=str(out_hyst))
        outputs.append(out_hyst)
        out_nm = self.output_dir / "hinge_nm.png"
        plot_hinge_nm_interaction(self.model, self.last, outfile=str(out_nm))
        outputs.append(out_nm)
        return outputs

    def plot_step(self, step: Optional[int] = None) -> List[Path]:
        if self.model is None or self.last is None:
            raise RuntimeError("Run a simulation before plotting results.")
        u_hist = self.last.get("u", np.array([], dtype=float))
        if u_hist.size == 0:
            raise RuntimeError("No displacement history available.")
        if step is None:
            idx = u_hist.shape[0] - 1
        else:
            idx = int(step)
            idx = max(0, min(idx, u_hist.shape[0] - 1))

        outputs = []
        for field in ("U", "S", "both"):
            out_path = self.output_dir / f"state_{idx}_{field}.png"
            fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
            plot_structure_state(
                ax,
                self.model,
                u_hist[idx],
                title=f"State {idx} ({field})",
                scale=1.0,
                field=field,
            )
            fig.savefig(out_path, dpi=170)
            plt.close(fig)
            outputs.append(out_path)
        return outputs

    def summary(self) -> str:
        if self.last is None:
            return "No results yet. Run a simulation first."
        t = self.last.get("t", np.array([], dtype=float))
        drift = self.last.get("drift", np.array([], dtype=float))
        max_drift = float(np.max(np.abs(drift))) if drift.size else float("nan")
        total_time = self.meta.get("total_time", float("nan"))
        return (
            f"Results summary\n"
            f"- total_time: {total_time:.3f}s\n"
            f"- steps: {t.size}\n"
            f"- peak_drift: {max_drift:.5f}\n"
        )


def _extreme_nodes_by_y(model) -> tuple[tuple[int, int], tuple[int, int], float]:
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


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  open <file>         Load an .inp file from the current directory\n"
        "  run                 Run the simulation\n"
        "  status on|off       Toggle per-increment status output\n"
        "  plot assembly       Plot the model assembly + boundary conditions\n"
        "  plot hinges         Plot hinge hysteresis (M-θ) and N-M interaction\n"
        "  plot step [index]   Plot U/S/both fields at a time step (default last)\n"
        "  results             Show results summary\n"
        "  help                Show this help\n"
        "  exit                Quit\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CLI for dinamica-computacional.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots (default: ./outputs/cli).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Enable per-increment status output during simulation.",
    )
    args = parser.parse_args()

    session = SimulationSession(output_dir=_outputs_dir(args.output_dir), verbose=args.status)
    print("dinamica-computacional CLI. Type 'help' for commands.")

    while True:
        try:
            raw = input("dc> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = shlex.split(raw)
        cmd = parts[0].lower()
        try:
            if cmd in {"exit", "quit"}:
                break
            if cmd == "help":
                _print_help()
            elif cmd == "open" and len(parts) > 1:
                session.load(parts[1])
                print(f"Loaded {parts[1]}")
            elif cmd == "run":
                session.run()
                print("Simulation completed.")
            elif cmd == "status" and len(parts) > 1:
                if parts[1].lower() in {"on", "true", "1"}:
                    session.verbose = True
                elif parts[1].lower() in {"off", "false", "0"}:
                    session.verbose = False
                print(f"Status output: {'on' if session.verbose else 'off'}")
            elif cmd == "plot" and len(parts) > 1:
                target = parts[1].lower()
                if target == "assembly":
                    path = session.plot_assembly()
                    print(f"Wrote {path}")
                elif target == "hinges":
                    paths = session.plot_hinges()
                    for path in paths:
                        print(f"Wrote {path}")
                elif target == "step":
                    idx = int(parts[2]) if len(parts) > 2 else None
                    paths = session.plot_step(idx)
                    for path in paths:
                        print(f"Wrote {path}")
                else:
                    print(f"Unknown plot target '{target}'.")
            elif cmd == "results":
                print(session.summary())
            else:
                print("Unknown command. Type 'help' for available commands.")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
