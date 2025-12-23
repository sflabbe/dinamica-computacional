"""Energy balance diagnostics and plots.

We use the standard power balance for second order systems:

    M a + C v + R_int(u) = P_ext(t)

Multiply by v and integrate in time:

    T(t) - T(0) + \\int v^T C v dt + \\int R_int \\cdot du = \\int P_ext \\cdot du

This module exports:
- anregung (excitation) time histories
- cumulative work terms and residual

Notes
-----
* These exports are *diagnostic*: they do not alter the solution.
* Numerical (algorithmic) damping depends on the chosen integrator (e.g. HHT-\u03b1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export_anregung(out: Path, prefix: str, t: np.ndarray, ag: np.ndarray) -> Path:
    """Save excitation (anregung) as csv and png."""
    out.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float).ravel()
    ag = np.asarray(ag, dtype=float).ravel()

    csv_path = out / f"{prefix}_anregung.csv"
    np.savetxt(csv_path, np.column_stack([t, ag]), delimiter=",", header="t,ag", comments="")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, ag)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("a_g [m/s^2]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = out / f"{prefix}_anregung.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    return png_path


def export_energy_balance(out: Path, prefix: str, t: np.ndarray, energy: Dict[str, np.ndarray]) -> Dict[str, Path]:
    """Save energy balance arrays as csv and png.

    Expected keys in `energy`:
      T, W_ext, W_int, W_damp, residual
    Optional:
      W_pl

    The exported CSV contains the cumulative works (integrated in time) as produced
    by the integrator.
    """
    out.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float).ravel()

    T = np.asarray(energy.get("T", np.zeros_like(t)), dtype=float).ravel()
    Wext = np.asarray(energy.get("W_ext", np.zeros_like(t)), dtype=float).ravel()
    Wint = np.asarray(energy.get("W_int", np.zeros_like(t)), dtype=float).ravel()
    Wd = np.asarray(energy.get("W_damp", np.zeros_like(t)), dtype=float).ravel()
    Wpl = np.asarray(energy.get("W_pl", np.zeros_like(t)), dtype=float).ravel() if "W_pl" in energy else None
    res = np.asarray(energy.get("residual", np.zeros_like(t)), dtype=float).ravel()

    denom = np.maximum(1e-12, np.maximum(np.abs(Wext), np.abs(T)))
    rel = res / denom

    cols = [t, T, Wext, Wint, Wd]
    header = "t,T,W_ext,W_int,W_damp"
    if Wpl is not None:
        cols.append(Wpl)
        header += ",W_pl"
    cols.extend([res, rel])
    header += ",residual,residual_rel"

    csv_path = out / f"{prefix}_energy_balance.csv"
    np.savetxt(csv_path, np.column_stack(cols), delimiter=",", header=header, comments="")

    # Plot 1: kinetic vs work-balance prediction
    fig, ax = plt.subplots(figsize=(6, 3.6))
    T0 = float(T[0]) if T.size else 0.0
    T_pred = T0 + (Wext - Wint - Wd)
    ax.plot(t, T, label="T (kinetic)")
    ax.plot(t, T_pred, "--", label="T_pred = T0+Wext-Wint-Wd")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Energy [J]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png_terms = out / f"{prefix}_energy_terms.png"
    fig.savefig(png_terms, dpi=160)
    plt.close(fig)

    # Plot 2: residual (absolute)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, res, label="residual")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Energy residual [J]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_res = out / f"{prefix}_energy_residual.png"
    fig.savefig(png_res, dpi=160)
    plt.close(fig)

    # Plot 3: residual (relative)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(t, rel)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("residual / max(|Wext|,|T|)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_rel = out / f"{prefix}_energy_residual_rel.png"
    fig.savefig(png_rel, dpi=160)
    plt.close(fig)

    # Plot 4: cumulative works overview
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(t, Wext, label="W_ext")
    ax.plot(t, Wint, label="W_int")
    ax.plot(t, Wd, label="W_damp")
    if Wpl is not None:
        ax.plot(t, Wpl, label="W_pl")
    ax.plot(t, res, "--", label="residual")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Cumulative work [J]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    png_works = out / f"{prefix}_energy_works.png"
    fig.savefig(png_works, dpi=160)
    plt.close(fig)

    return {"csv": csv_path, "terms": png_terms, "residual": png_res, "residual_rel": png_rel, "works": png_works}
