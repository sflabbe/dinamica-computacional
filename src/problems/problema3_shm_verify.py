"""Problema 3: Reproducción/chequeo de la Fig. 4 (Sivaselvan & Reinhorn, 2000).

Este script genera curvas M–θ (análogo fuerza–desplazamiento) para tres especímenes de
conexiones (SAC Joint Venture 1996) usando los parámetros reportados en la Fig. 4:

- α (stiffness degradation, Eq. 6)
- β1, β2 (strength degradation, Eq. 8)
- μ_ult (ductilidad última)

Además, plotea las histeresis con un gradiente de color en el tiempo (step) para facilitar
la lectura de los ciclos.
"""

from __future__ import annotations

# --- bootstrap para ejecutar el script desde la raíz de la repo ---
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dc_solver.hinges.models import SHMSivaselvanReinhorn1D
from dc_solver.post.plotting import plot_hysteresis_time_gradient


def _outputs_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cycle_path(amplitude: float, steps_per_half: int = 60) -> np.ndarray:
    """Un ciclo completo 0→+A→0→−A→0 con discretización uniforme."""
    up = np.linspace(0.0, amplitude, steps_per_half, endpoint=False)
    down = np.linspace(amplitude, 0.0, steps_per_half, endpoint=False)
    neg = np.linspace(0.0, -amplitude, steps_per_half, endpoint=False)
    back = np.linspace(-amplitude, 0.0, steps_per_half + 1)
    return np.concatenate([up, down, neg, back])


def _history_from_cycles(amplitudes: list[float], steps_per_half: int = 60) -> np.ndarray:
    series: list[float] = []
    for amp in amplitudes:
        cycle = _cycle_path(float(amp), steps_per_half=steps_per_half)
        if series:
            cycle = cycle[1:]  # evitar duplicar el punto de unión
        series.extend(cycle.tolist())
    return np.asarray(series, dtype=float)


@dataclass(frozen=True)
class Fig4Specimen:
    name: str
    alpha: float
    beta1: float
    beta2: float
    mu_ult: float


FIG4_SPECIMENS = [
    Fig4Specimen("EERC-RN3", alpha=10.0, beta1=0.60, beta2=0.30, mu_ult=8.0),
    Fig4Specimen("UCSD-1R",  alpha=5.0,  beta1=0.40, beta2=0.20, mu_ult=7.1),
    Fig4Specimen("UTA-3R",   alpha=4.0,  beta1=0.20, beta2=0.30, mu_ult=5.6),
]


def _run_specimen(spec: Fig4Specimen, theta_hist: np.ndarray, *, K0: float, My0: float):
    hinge = SHMSivaselvanReinhorn1D(
        K0=K0,
        My0=My0,
        a_post=0.02,          # ratio post-yield (latin a), no reportado en Fig. 4
        N_smooth=10.0,        # suavidad transición (N), no reportado en Fig. 4
        eta1=0.5, eta2=0.5,   # forma descarga (η1,η2), no reportado en Fig. 4
        alpha_pivot=spec.alpha,
        beta1=spec.beta1,
        beta2=spec.beta2,
        mu_ult=spec.mu_ult,
        pinch=0.0,            # pinching no incluido en Fig. 4 → se deja apagado
    )

    n = theta_hist.size
    M = np.zeros(n)
    Ktan = np.zeros(n)
    H = np.zeros(n)
    th_max_pos = np.zeros(n)
    th_max_neg = np.zeros(n)
    My_pos = np.zeros(n)
    My_neg = np.zeros(n)

    th_prev = theta_hist[0]

    for i in range(n):
        th = float(theta_hist[i])
        dth = th - float(th_prev) if i > 0 else th

        M_new, k_tan, th_new, H_new, M_comm = hinge.eval_increment(dth)

        # commit
        hinge.th_comm = th_new
        hinge.M_comm = M_comm
        hinge.H_comm = H_new
        hinge.th_max_pos_comm = max(hinge.th_max_pos_comm, th_new)
        hinge.th_max_neg_comm = min(hinge.th_max_neg_comm, th_new)

        M[i] = M_new
        Ktan[i] = k_tan
        H[i] = H_new
        th_max_pos[i] = hinge.th_max_pos_comm
        th_max_neg[i] = hinge.th_max_neg_comm
        My_pos[i] = float(hinge.My_pos_comm or 0.0)
        My_neg[i] = float(hinge.My_neg_comm or 0.0)

        th_prev = th

    return {
        "theta": theta_hist,
        "M": M,
        "Ktan": Ktan,
        "H": H,
        "th_max_pos": th_max_pos,
        "th_max_neg": th_max_neg,
        "My_pos": My_pos,
        "My_neg": My_neg,
    }


def main() -> None:
    out = _outputs_dir()

    # Escala base (puedes cambiarla; la forma depende principalmente de α, β1, β2, μ_ult)
    K0 = 2.5e8
    My0 = 3.0e5

    th_y0 = My0 / K0  # rotación/curvatura de fluencia
    duct_levels = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    results = {}

    # Ejecutar y guardar CSV por espécimen
    for spec in FIG4_SPECIMENS:
        # Amplitudes hasta ~0.95 μ_ult para evitar cortar exactamente en el límite
        amps = [mu * th_y0 for mu in duct_levels if mu <= 0.95 * spec.mu_ult]
        theta_hist = _history_from_cycles(amps, steps_per_half=70)

        res = _run_specimen(spec, theta_hist, K0=K0, My0=My0)
        results[spec.name] = (spec, res)

        csv_path = out / f"problem3_fig4_{spec.name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "theta", "M", "Ktan", "H", "th_max_pos", "th_max_neg", "My_pos", "My_neg"])
            for i in range(theta_hist.size):
                w.writerow([
                    i,
                    float(res["theta"][i]),
                    float(res["M"][i]),
                    float(res["Ktan"][i]),
                    float(res["H"][i]),
                    float(res["th_max_pos"][i]),
                    float(res["th_max_neg"][i]),
                    float(res["My_pos"][i]),
                    float(res["My_neg"][i]),
                ])

    # Figura tipo Fig. 4: tres histeresis (analíticas) con gradiente temporal
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.2, 10.5), sharex=False)
    lcs = []

    for ax, (name, (spec, res)) in zip(axes, results.items()):
        theta = res["theta"]
        M = res["M"]
        t = np.linspace(0.0, 1.0, num=theta.size)

        lc = plot_hysteresis_time_gradient(
            theta, M, t,
            ax=ax,
            cmap="plasma",
            lw=2.8,
            add_colorbar=False,
            cbar_label="t (normalizado)",
        )
        lcs.append(lc)

        ax.set_title(
            f"{spec.name}   α={spec.alpha:g}, β₁={spec.beta1:g}, β₂={spec.beta2:g}, μ₍ult₎={spec.mu_ult:g}",
            fontsize=11,
        )
        ax.set_xlabel("theta")
        ax.set_ylabel("M [N-m]")
        ax.grid(True, alpha=0.25)

    # Colorbar global (usa el último lc, mismo norm 0..1)
    cbar = fig.colorbar(lcs[-1], ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("tiempo / step (normalizado)")

    fig.tight_layout()
    fig.savefig(out / "problem3_fig4_specimens.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
