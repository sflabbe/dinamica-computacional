from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dc_solver.modal.eigensolver import condense_massless_dofs, solve_eigenpairs


@dataclass
class ModalResults:
    omega: np.ndarray
    freq_hz: np.ndarray
    periods: np.ndarray
    modes_free: np.ndarray
    modes_full: np.ndarray
    free_dofs: np.ndarray
    participation_factors: np.ndarray
    effective_modal_mass: np.ndarray
    effective_modal_mass_ratio: np.ndarray
    cumulative_mass_ratio: np.ndarray
    direction: str
    meta: dict[str, object]


def run_modal_analysis(model, n_modes=6, *, direction="ux", u_ref=None, mass_tol=1e-12) -> ModalResults:
    fd = model.free_dofs()
    if u_ref is None:
        u_ref = np.zeros(model.ndof(), dtype=float)

    K_ff, _, _ = model.assemble(u_ref, u_ref)
    M_diag_ff = np.asarray(model.mass_diag[fd], dtype=float)
    condensed = condense_massless_dofs(K_ff, M_diag_ff, mass_tol=mass_tol)

    omega, phi_a = solve_eigenpairs(condensed.K_eff, condensed.M_eff, n_modes=n_modes)
    modes_free = condensed.T_expand @ phi_a

    modes_full = np.zeros((model.ndof(), modes_free.shape[1]), dtype=float)
    modes_full[fd, :] = modes_free

    M_full = np.diag(np.asarray(model.mass_diag, dtype=float))
    for i in range(modes_full.shape[1]):
        nrm = float(modes_full[:, i].T @ M_full @ modes_full[:, i])
        modes_full[:, i] /= np.sqrt(nrm)
        modes_free[:, i] /= np.sqrt(nrm)

    axis = 0 if direction == "ux" else 1 if direction == "uy" else None
    if axis is None:
        raise ValueError("direction must be 'ux' or 'uy'.")
    r = np.zeros(model.ndof(), dtype=float)
    for node in model.nodes:
        r[node.dof_u[axis]] = 1.0

    L = np.array([modes_full[:, i].T @ M_full @ r for i in range(modes_full.shape[1])], dtype=float)
    eff_mass = L**2
    total_mass = float(r.T @ M_full @ r)
    ratio = eff_mass / total_mass if total_mass > 0 else np.zeros_like(eff_mass)
    cum = np.cumsum(ratio)

    freq_hz = omega / (2.0 * np.pi)
    periods = np.divide(1.0, freq_hz, out=np.full_like(freq_hz, np.inf), where=freq_hz > 0)

    return ModalResults(
        omega=omega,
        freq_hz=freq_hz,
        periods=periods,
        modes_free=modes_free,
        modes_full=modes_full,
        free_dofs=fd,
        participation_factors=L,
        effective_modal_mass=eff_mass,
        effective_modal_mass_ratio=ratio,
        cumulative_mass_ratio=cum,
        direction=direction,
        meta={"warnings": condensed.warnings, "active_dofs_local": condensed.active_dofs_local, "massless_dofs_local": condensed.massless_dofs_local},
    )
