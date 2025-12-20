"""Problema 3: Verificación SHM con historia tipo Figura 4."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dc_solver.hinges.models import SHMBeamHinge1D
from dc_solver.reporting.run_info import build_run_info, write_run_info


def _outputs_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cycle_path(amplitude: float, steps_per_half: int = 40) -> np.ndarray:
    up = np.linspace(0.0, amplitude, steps_per_half, endpoint=False)
    down = np.linspace(amplitude, 0.0, steps_per_half, endpoint=False)
    neg = np.linspace(0.0, -amplitude, steps_per_half, endpoint=False)
    back = np.linspace(-amplitude, 0.0, steps_per_half + 1)
    return np.concatenate([up, down, neg, back])


def _history_from_cycles(amplitudes, steps_per_half: int = 40) -> np.ndarray:
    series = []
    for amp in amplitudes:
        cycle = _cycle_path(float(amp), steps_per_half=steps_per_half)
        if series:
            cycle = cycle[1:]
        series.extend(cycle.tolist())
    return np.asarray(series, dtype=float)


def _load_reference(path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return None
    return np.asarray(data["theta"], dtype=float), np.asarray(data["M"], dtype=float)


def main() -> None:
    out = _outputs_dir()

    theta_hist = _history_from_cycles([0.002, 0.004, 0.006, 0.008], steps_per_half=60)

    hinge = SHMBeamHinge1D(
        K0_0=2.5e8,
        My_0=3.0e5,
        alpha_post=0.02,
        cK=1.8,
        cMy=1.2,
        bw_beta=0.7,
        bw_gamma=0.3,
        bw_n=2.0,
        pinch=0.35,
        theta_pinch=0.0015,
    )

    steps = theta_hist.size
    M = np.zeros(steps)
    Ktan = np.zeros(steps)
    th_hist = np.zeros(steps)
    a_hist = np.zeros(steps)

    th_prev = theta_hist[0]

    for i in range(steps):
        th = theta_hist[i]
        dth = th - th_prev if i > 0 else th

        M_new, k_tan, th_new, a_new, M_comm = hinge.eval_increment(dth)
        hinge.th_comm = th_new
        hinge.a_comm = a_new
        hinge.M_comm = M_comm

        M[i] = M_new
        Ktan[i] = k_tan
        th_hist[i] = th_new
        a_hist[i] = a_new
        th_prev = th

    csv_path = out / "problem3_shm_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "theta", "M", "a", "th", "Ktan"])
        for i in range(steps):
            writer.writerow([i, theta_hist[i], M[i], a_hist[i], th_hist[i], Ktan[i]])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(theta_hist, M, label="SHM model")

    ref_path = out / "problem3_fig4_reference.csv"
    ref = _load_reference(ref_path)
    if ref is not None:
        theta_ref, M_ref = ref
        ax.plot(theta_ref, M_ref, "k--", label="reference")
        sort_idx = np.argsort(theta_hist)
        M_interp = np.interp(theta_ref, theta_hist[sort_idx], M[sort_idx])
        rmse = math.sqrt(float(np.mean((M_interp - M_ref) ** 2)))
        ax.text(0.05, 0.95, f"RMSE={rmse:.3e}", transform=ax.transAxes, va="top")

    ax.set_xlabel("theta")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "problem3_shm_m_theta.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(th_hist, a_hist)
    ax.set_xlabel("theta")
    ax.set_ylabel("Dissipated energy proxy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "problem3_shm_energy.png", dpi=160)
    plt.close(fig)

    # Run-info export
    info = build_run_info(
        job="problem3_shm_verify",
        output_dir=str(out),
        meta={
            "n_steps": int(steps),
            "theta": {"max": float(np.max(theta_hist)), "min": float(np.min(theta_hist))},
            "hinge": {
                "K0_0": float(hinge.K0_0),
                "My_0": float(hinge.My_0),
                "alpha_post": float(hinge.alpha_post),
                "cK": float(hinge.cK),
                "cMy": float(hinge.cMy),
                "bw_beta": float(hinge.bw_beta),
                "bw_gamma": float(hinge.bw_gamma),
                "bw_n": float(hinge.bw_n),
                "pinch": float(hinge.pinch),
                "theta_pinch": float(hinge.theta_pinch),
            },
        },
    )
    write_run_info(out, base_name="problem3_runinfo", info=info)


if __name__ == "__main__":
    main()
