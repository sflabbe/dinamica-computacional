"""Problema 2: Interacción N–M por poligonal y verificación cíclica."""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Allow running as:
#   python src/problems/problema2_interaccion.py
# without requiring editable installs.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from dc_solver.post.hysteresis_gradient import add_time_gradient_line, add_colorbar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from plastic_hinge import PlasticHingeNM
from problems.problema2_secciones_nm import (
    compute_interaction_polygon,
    make_section_S1,
    make_section_S2,
)


def _solve_deps_for_N_control(
    hinge: PlasticHingeNM,
    dtheta: float,
    N_target: float,
    deps_prev: float = 0.0,
    N_tol: float = 1e3,
    deps_cap: float = 2e-3,
    max_expand: int = 12,
    max_bisect: int = 30,
) -> float:
    """Find axial increment dε such that updated N ≈ N_target.

    Robust scalar root solve using bracket expansion + bisection.
    The hinge state is NOT modified during evaluations (commit=False).

    Notes
    -----
    Newton-type methods are fragile here because the polygonal return-mapping
    makes N(dε) non-smooth when the active face set changes.
    """

    def f(deps: float) -> float:
        info = hinge.update(np.array([deps, dtheta], float), commit=False)
        return float(info["s"][0] - N_target)

    # Initial guess (continuation)
    deps0 = float(np.clip(deps_prev, -deps_cap, deps_cap))
    f0 = f(deps0)
    if abs(f0) <= N_tol:
        return deps0

    # Expand a symmetric bracket around deps0
    step = max(1e-10, 0.25 * abs(deps0) + 1e-8)
    best_deps = deps0
    best_abs = abs(f0)
    a = b = deps0
    fa = fb = f0
    bracket_found = False

    for _ in range(max_expand):
        a = float(np.clip(deps0 - step, -deps_cap, deps_cap))
        b = float(np.clip(deps0 + step, -deps_cap, deps_cap))
        fa = f(a)
        fb = f(b)

        if abs(fa) < best_abs:
            best_abs = abs(fa)
            best_deps = a
        if abs(fb) < best_abs:
            best_abs = abs(fb)
            best_deps = b

        if fa == 0.0:
            return a
        if fb == 0.0:
            return b

        if fa * fb < 0.0:
            bracket_found = True
            break

        # If we hit the cap and still no sign change, no point expanding further
        if abs(a) >= deps_cap and abs(b) >= deps_cap:
            break
        step *= 2.0

    if not bracket_found:
        # No clean bracket (non-monotone / saturated). Return best we saw.
        return float(best_deps)

    # Bisection
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= N_tol:
            return float(mid)
        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return float(0.5 * (lo + hi))


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
    hinge = PlasticHingeNM(surface=surface, K=K, enable_substepping=True)
    return hinge, KN, KM


def _simulate_history(
    hinge: PlasticHingeNM,
    q_hist: np.ndarray,
    *,
    axial_control: bool = False,
    N_target: float = 0.0,
    N_tol: float = 1e3,
) -> Dict[str, np.ndarray]:
    # Copy because in axial_control mode we overwrite the solved epsilon history
    q_hist = np.array(q_hist, dtype=float, copy=True)
    dq_hist = np.diff(q_hist, axis=0)
    n_steps = dq_hist.shape[0]

    s = np.zeros((n_steps, 2))
    s_trial = np.zeros((n_steps, 2))
    q_p = np.zeros((n_steps, 2))
    dq_p_inc = np.zeros((n_steps, 2))
    active = np.zeros(n_steps, dtype=int)
    max_violation = np.zeros(n_steps)

    # For axial_control (flexion pura): solve dε per step so that N ≈ N_target.
    eps = float(q_hist[0, 0])
    deps_prev = 0.0

    for i, dq in enumerate(dq_hist):
        dq = np.asarray(dq, float)
        if axial_control:
            # Flexión pura (control axial): prescribir dtheta y resolver dε
            # tal que N ≈ N_target en el estado *corregido* (con return mapping).
            #
            # Importante: evitar Newton directo; la proyección poligonal hace que
            # N(dε) no sea suave cuando cambia el set activo.
            dtheta = float(dq[1])
            deps = _solve_deps_for_N_control(
                hinge,
                dtheta=dtheta,
                N_target=N_target,
                deps_prev=deps_prev,
                N_tol=N_tol,
                deps_cap=2e-3,
            )
            info = hinge.update(np.array([deps, dtheta], float), commit=True)

            deps_prev = float(deps)
            eps += float(deps)
            q_hist[i + 1, 0] = eps
        else:
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



def _plot_paths_gradient(out: Path, tag: str, surface, sims: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Export N–M paths with a time/step color gradient (one figure per case)."""
    verts = np.vstack([surface.vertices, surface.vertices[0]])
    for name, data in sims.items():
        s = data["s"]
        s_trial = data["s_trial"]
        steps = np.arange(s.shape[0], dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(verts[:, 0], verts[:, 1], "k-", lw=1.5, label="Yield hull")
        ax.plot(s_trial[:, 0], s_trial[:, 1], "--", alpha=0.25, lw=1.0, label="trial")
        lc = add_time_gradient_line(ax, s[:, 0], s[:, 1], c=steps, lw=2.0, alpha=1.0)
        add_colorbar(lc, ax, label="step")

        ax.set_xlabel("N [N]")
        ax.set_ylabel("M [N-m]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"problem2_interaction_{tag}_paths_{name}_gradient.png", dpi=170)
        plt.close(fig)


def _plot_hysteresis_gradient(out: Path, tag: str, sims: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Export hysteresis with a time/step color gradient (one figure per case)."""
    for name, data in sims.items():
        q = data["q"]
        s = data["s"]
        steps = np.arange(q.shape[0], dtype=float)
        norm = Normalize(vmin=float(steps.min()), vmax=float(steps.max()))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_mth, ax_neps = axes

        lc1 = add_time_gradient_line(ax_mth, q[:, 1], s[:, 1], c=steps, norm=norm, lw=2.0)
        ax_mth.set_xlabel("theta")
        ax_mth.set_ylabel("M [N-m]")
        ax_mth.grid(True, alpha=0.3)

        lc2 = add_time_gradient_line(ax_neps, q[:, 0], s[:, 0], c=steps, norm=norm, lw=2.0)
        ax_neps.set_xlabel("epsilon")
        ax_neps.set_ylabel("N [N]")
        ax_neps.grid(True, alpha=0.3)

        # Single shared colorbar (attach to right axis)
        add_colorbar(lc2, ax_neps, label="step")

        fig.suptitle(f"Problem 2 {tag} — {name} (time gradient)", y=1.02)
        fig.tight_layout()
        fig.savefig(out / f"problem2_interaction_{tag}_hysteresis_{name}_gradient.png", dpi=170)
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

    histories = _build_histories()

    for tag, section in ("S1", sec_s1), ("S2", sec_s2):
        hinge, _, _ = _make_hinge(section)
        sims: Dict[str, Dict[str, np.ndarray]] = {}
        for name, q_hist in histories.items():
            hinge.s = np.zeros(2, float)
            hinge.q_p = np.zeros(2, float)
            sims[name] = _simulate_history(
                hinge,
                q_hist,
                axial_control=(name == "flexion"),
                N_target=0.0,
                N_tol=1e3,
            )

        _plot_paths(out, tag, hinge.surface, sims)
        _plot_paths_gradient(out, tag, hinge.surface, sims)
        _plot_hysteresis(out, tag, sims)
        _plot_hysteresis_gradient(out, tag, sims)

        max_violation = max(float(np.max(sim["max_violation"])) for sim in sims.values())
        summary = [
            f"problem2_interaction_{tag}",
            f"max_violation={max_violation:.3e}",
        ]
        (out / f"problem2_interaction_{tag}_summary.txt").write_text("\n".join(summary), encoding="utf-8")


if __name__ == "__main__":
    main()