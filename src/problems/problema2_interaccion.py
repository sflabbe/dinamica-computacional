"""Problema 2: Interacción N–M por poligonal y verificación cíclica."""

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


def _build_histories() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    eps_history = _history_from_cycles([2e-4, 4e-4, 6e-4], steps_per_half=50)
    th_history = _history_from_cycles([0.002, 0.004, 0.006], steps_per_half=50)

    t = np.linspace(0.0, 6.0 * math.pi, 400)
    eps_comb = 2.5e-4 * np.sin(t)
    th_comb = 0.007 * np.sin(2.0 * t + 0.3)

    # NOTE:
    # - "axial": prescribe ε history, θ=0
    # - "combined": prescribe ε and θ
    # - "flexion": *pure bending with N≈0*, so ε must be solved each step (force-control),
    #   not imposed as ε=0 (which creates a large reaction N and a serrucho in M–θ).
    histories = {
        "axial": np.column_stack([eps_history, np.zeros_like(eps_history)]),
        "combined": np.column_stack([eps_comb, th_comb]),
    }
    return histories, th_history


def _make_hinge(section) -> Tuple[PlasticHingeNM, float, float]:
    surface = compute_interaction_polygon(section, symmetric_M=True, symmetric_N=False, n=90)
    Lp = 0.5 * section.h
    E = 30e9
    I = section.b * section.h**3 / 12.0
    KN = E * section.Ac / Lp
    KM = E * I / Lp
    K = np.diag([KN, KM])
    hinge = PlasticHingeNM(surface=surface, K=K, enable_substepping=True)
    return hinge, KN, KM


def _simulate_history(hinge: PlasticHingeNM, q_hist: np.ndarray) -> Dict[str, np.ndarray]:
    q_hist = np.asarray(q_hist, dtype=float)
    dq_hist = np.diff(q_hist, axis=0)
    n_steps = dq_hist.shape[0]

    s = np.zeros((n_steps, 2))
    s_trial = np.zeros((n_steps, 2))
    q_p = np.zeros((n_steps, 2))
    dq_p_inc = np.zeros((n_steps, 2))
    active = np.zeros(n_steps, dtype=int)
    max_violation = np.zeros(n_steps)

    for i, dq in enumerate(dq_hist):
        info = hinge.update(dq)
        s[i] = info["s"]
        s_trial[i] = info["s_trial"]
        dq_p_inc[i] = info["dq_p_inc"]
        q_p[i] = info["q_p"]
        active[i] = int(info["active"].size)
        max_violation[i] = float(np.max(hinge.surface.A @ s[i] - hinge.surface.b))

    return {
        "q": q_hist[1:],
        "s": s,
        "s_trial": s_trial,
        "q_p": q_p,
        "dq_p_inc": dq_p_inc,
        "active": active,
        "max_violation": max_violation,
    }



def _simulate_flexion_control_N(
    hinge: PlasticHingeNM,
    theta_hist: np.ndarray,
    N_target: float = 0.0,
    N_tol: float = 1.0e3,
    max_iter: int = 25,
) -> Dict[str, np.ndarray]:
    """Pure flexion (Problema 2): prescribe θ(t) and solve ε increment so that N≈N_target (default 0).

    Setting ε(t)=0 is axial restraint and generates large reaction N.
    In an N–M coupled hinge this alters My(N) and produces a 'serrucho' artifact in M–θ.
    This routine does proper axial force-control using a local Newton iteration on dε.
    """
    theta_hist = np.asarray(theta_hist, dtype=float).reshape(-1)
    dth_hist = np.diff(theta_hist)
    n_steps = dth_hist.shape[0]

    q = np.zeros((n_steps, 2))
    s = np.zeros((n_steps, 2))
    s_trial = np.zeros((n_steps, 2))
    q_p = np.zeros((n_steps, 2))
    dq_p_inc = np.zeros((n_steps, 2))
    active = np.zeros(n_steps, dtype=int)
    max_violation = np.zeros(n_steps)

    eps = 0.0
    th = float(theta_hist[0])

    for i, dth in enumerate(dth_hist):
        deps = 0.0
        for _ in range(max_iter):
            info_t = hinge.update(np.array([deps, dth], dtype=float), commit=False)
            N = float(info_t["s"][0])
            r = N - float(N_target)
            if abs(r) <= float(N_tol):
                break
            Kt = info_t.get("Kt", hinge.K)
            dNdEps = float(Kt[0, 0])
            if not np.isfinite(dNdEps) or abs(dNdEps) < 1e-12:
                dNdEps = float(hinge.K[0, 0])
            deps -= r / dNdEps

        info = hinge.update(np.array([deps, dth], dtype=float), commit=True)

        eps += deps
        th += dth
        q[i, :] = [eps, th]

        s[i] = info["s"]
        s_trial[i] = info["s_trial"]
        dq_p_inc[i] = info["dq_p_inc"]
        q_p[i] = info["q_p"]
        active[i] = int(info["active"].size)
        max_violation[i] = float(np.max(hinge.surface.A @ s[i] - hinge.surface.b))

    return {
        "q": q,
        "s": s,
        "s_trial": s_trial,
        "q_p": q_p,
        "dq_p_inc": dq_p_inc,
        "active": active,
        "max_violation": max_violation,
    }



def _plot_paths(out: Path, tag: str, surface, sims: Dict[str, Dict[str, np.ndarray]]) -> None:
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
    fig.savefig(out / f"problem2_interaction_{tag}_paths.png", dpi=160)
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
    fig.savefig(out / f"problem2_interaction_{tag}_hysteresis.png", dpi=160)
    plt.close(fig)


def main() -> None:
    out = _outputs_dir()

    sec_s1 = make_section_S1()
    sec_s2 = make_section_S2()

    surf_s1 = compute_interaction_polygon(sec_s1, symmetric_M=True, symmetric_N=False, n=90)
    surf_s2 = compute_interaction_polygon(sec_s2, symmetric_M=True, symmetric_N=False, n=90)

    for tag, surface in ("S1", surf_s1), ("S2", surf_s2):
        verts = np.vstack([surface.vertices, surface.vertices[0]])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5)
        ax.set_xlabel("N [N]")
        ax.set_ylabel("M [N-m]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"problem2_interaction_{tag}_yield_hull.png", dpi=160)
        plt.close(fig)

    histories, th_flex = _build_histories()

    for tag, section in ("S1", sec_s1), ("S2", sec_s2):
        hinge, _, _ = _make_hinge(section)
        sims: Dict[str, Dict[str, np.ndarray]] = {}
        for name, q_hist in histories.items():
            hinge.s = np.zeros(2, float)
            hinge.q_p = np.zeros(2, float)
            sims[name] = _simulate_history(hinge, q_hist)

        # Pure flexion with axial force control: keep N≈0 (no axial restraint)
        hinge.s = np.zeros(2, float)
        hinge.q_p = np.zeros(2, float)
        sims["flexion"] = _simulate_flexion_control_N(hinge, th_flex, N_target=0.0)

        _plot_paths(out, tag, hinge.surface, sims)
        _plot_hysteresis(out, tag, sims)

        max_violation = max(float(np.max(sim["max_violation"])) for sim in sims.values())
        summary = [
            f"problem2_interaction_{tag}",
            f"max_violation={max_violation:.3e}",
        ]
        (out / f"problem2_interaction_{tag}_summary.txt").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
