"""Problema 2: Verificación de rótula elasto-plástica N-M."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plastic_hinge import PlasticHingeNM
from problems.problema2_secciones_nm import (
    compute_interaction_polygon,
    make_section_S1,
    make_section_S2,
)


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


def _history_from_cycles(amplitudes: Iterable[float], steps_per_half: int = 40) -> np.ndarray:
    series: List[float] = []
    for amp in amplitudes:
        cycle = _cycle_path(float(amp), steps_per_half=steps_per_half)
        if series:
            cycle = cycle[1:]
        series.extend(cycle.tolist())
    return np.asarray(series, dtype=float)


def _build_histories() -> Dict[str, np.ndarray]:
    eps_history = _history_from_cycles([2e-4, 4e-4, 6e-4], steps_per_half=50)
    th_history = _history_from_cycles([0.002, 0.004, 0.006], steps_per_half=50)

    t = np.linspace(0.0, 6.0 * math.pi, 400)
    eps_comb = 2.5e-4 * np.sin(t)
    th_comb = 0.007 * np.sin(2.0 * t + 0.3)

    histories = {
        "axial": np.column_stack([eps_history, np.zeros_like(eps_history)]),
        "flexion": np.column_stack([np.zeros_like(th_history), th_history]),
        "combined": np.column_stack([eps_comb, th_comb]),
    }
    return histories


def _make_hinge(section) -> Tuple[PlasticHingeNM, float, float]:
    surface = compute_interaction_polygon(section, symmetric_M=True, symmetric_N=False, n=90)
    Lp = 0.5 * section.h
    E = 30e9
    I = section.b * section.h**3 / 12.0
    KN = E * section.Ac / Lp
    KM = E * I / Lp
    K = np.diag([KN, KM])
    return PlasticHingeNM(surface=surface, K=K), KN, KM


def _simulate_history(hinge: PlasticHingeNM, q_hist: np.ndarray) -> Dict[str, np.ndarray]:
    q_hist = np.asarray(q_hist, dtype=float)
    dq_hist = np.diff(q_hist, axis=0)
    n_steps = dq_hist.shape[0]

    s = np.zeros((n_steps, 2))
    s_trial = np.zeros((n_steps, 2))
    q_p = np.zeros((n_steps, 2))
    dq_p_inc = np.zeros((n_steps, 2))
    flow_check = np.zeros((n_steps, 2))
    active_count = np.zeros(n_steps, dtype=int)
    lam_norm = np.zeros(n_steps, dtype=float)

    for i, dq in enumerate(dq_hist):
        info = hinge.update(dq)
        s[i] = info["s"]
        s_trial[i] = info["s_trial"]
        dq_p_inc[i] = info["dq_p_inc"]
        q_p[i] = info["q_p"]
        flow_check[i] = info.get("flow_check", np.zeros(2))
        active = info.get("active", np.zeros((0,), dtype=int))
        lam = info.get("lam", np.zeros((0,)))
        active_count[i] = int(active.size)
        lam_norm[i] = float(np.linalg.norm(lam)) if lam.size else 0.0

    return {
        "q": q_hist[1:],
        "s": s,
        "s_trial": s_trial,
        "q_p": q_p,
        "dq_p_inc": dq_p_inc,
        "flow_check": flow_check,
        "active_count": active_count,
        "lam_norm": lam_norm,
    }


def _evaluate_checks(surface, sim_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    s = sim_data["s"]
    dq_p_inc = sim_data["dq_p_inc"]
    flow_check = sim_data["flow_check"]
    active_count = sim_data["active_count"]

    inside = np.array([surface.is_inside(si, tol=1e-8) for si in s])
    inside_fail = int(np.size(inside) - int(np.sum(inside)))

    flow_errors: List[float] = []
    for i in range(s.shape[0]):
        if active_count[i] == 0:
            continue
        denom = np.linalg.norm(dq_p_inc[i])
        if denom <= 0.0:
            continue
        err = np.linalg.norm(dq_p_inc[i] - flow_check[i]) / denom
        flow_errors.append(float(err))

    if flow_errors:
        flow_max = float(np.max(flow_errors))
        flow_mean = float(np.mean(flow_errors))
    else:
        flow_max = 0.0
        flow_mean = 0.0

    dWp = np.einsum("ij,ij->i", s, dq_p_inc)
    dWp_min = float(np.min(dWp)) if dWp.size else 0.0

    return {
        "inside_fail": inside_fail,
        "flow_max": flow_max,
        "flow_mean": flow_mean,
        "dWp_min": dWp_min,
    }


def _plot_paths(
    out: Path,
    tag: str,
    surface,
    sims: Dict[str, Dict[str, np.ndarray]],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    verts = np.vstack([surface.vertices, surface.vertices[0]])
    ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5, label="Yield hull")
    for name, data in sims.items():
        s = data["s"]
        s_trial = data["s_trial"]
        ax.plot(s_trial[:, 0], s_trial[:, 1], "--", alpha=0.4, label=f"{name} trial")
        ax.plot(s[:, 0], s[:, 1], "-", alpha=0.8, label=f"{name} corrected")
    ax.set_xlabel("N [N]")
    ax.set_ylabel("M [N-m]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"problem2_hinge_{tag}_paths.png", dpi=160)
    plt.close(fig)


def _plot_hysteresis(out: Path, tag: str, sims: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_mth, ax_neps = axes

    for name, data in sims.items():
        q = data["q"]
        s = data["s"]
        ax_mth.plot(q[:, 1], s[:, 1], label=name)
        ax_neps.plot(q[:, 0], s[:, 0], label=name)

    ax_mth.set_xlabel("theta")
    ax_mth.set_ylabel("M [N-m]")
    ax_mth.grid(True, alpha=0.3)

    ax_neps.set_xlabel("epsilon")
    ax_neps.set_ylabel("N [N]")
    ax_neps.grid(True, alpha=0.3)

    ax_mth.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"problem2_hinge_{tag}_hysteresis.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out = _outputs_dir()
    histories = _build_histories()

    results = []

    for tag, section_builder in [("S1", make_section_S1), ("S2", make_section_S2)]:
        section = section_builder()
        hinge, _, _ = _make_hinge(section)
        surface = hinge.surface

        sims: Dict[str, Dict[str, np.ndarray]] = {}
        for name, q_hist in histories.items():
            hinge = PlasticHingeNM(surface=surface, K=hinge.K.copy())
            sim = _simulate_history(hinge, q_hist)
            sims[name] = sim
            checks = _evaluate_checks(surface, sim)
            checks["section"] = tag
            checks["history"] = name
            results.append(checks)

        _plot_paths(out, tag, surface, sims)
        _plot_hysteresis(out, tag, sims)

    lines = [
        "Problem 2 hinge verification checks",
        "section,history,inside_fail,flow_max,flow_mean,dWp_min",
    ]
    for row in results:
        lines.append(
            f"{row['section']},{row['history']},{row['inside_fail']}"
            f",{row['flow_max']:.3e},{row['flow_mean']:.3e},{row['dWp_min']:.3e}"
        )

    (out / "problem2_hinge_checks.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
