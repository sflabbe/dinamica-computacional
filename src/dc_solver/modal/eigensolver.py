from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh


@dataclass
class CondensedSystem:
    free_dofs: np.ndarray
    active_dofs_local: np.ndarray
    massless_dofs_local: np.ndarray
    K_eff: np.ndarray
    M_eff: np.ndarray
    T_expand: np.ndarray
    warnings: list[str]


def _validate_symmetric(name: str, A: np.ndarray, tol: float = 1e-10) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {A.shape}.")
    if not np.allclose(A, A.T, atol=tol, rtol=0.0):
        raise ValueError(f"{name} must be symmetric within tolerance {tol}.")


def condense_massless_dofs(K_ff, M_diag_ff, *, mass_tol=1e-12, reg=1e-14) -> CondensedSystem:
    K_ff = np.asarray(K_ff, dtype=float)
    M_diag_ff = np.asarray(M_diag_ff, dtype=float)
    _validate_symmetric("K_ff", K_ff)
    if M_diag_ff.ndim != 1 or M_diag_ff.shape[0] != K_ff.shape[0]:
        raise ValueError("M_diag_ff must be a vector with same length as K_ff dimension.")

    free_dofs = np.arange(K_ff.shape[0], dtype=int)
    active = np.flatnonzero(M_diag_ff > mass_tol)
    massless = np.flatnonzero(M_diag_ff <= mass_tol)
    warnings: list[str] = []

    if active.size == 0:
        raise ValueError("No active DOFs with positive mass after applying mass_tol.")

    if massless.size == 0:
        T_expand = np.eye(K_ff.shape[0])
        return CondensedSystem(free_dofs, active, massless, K_ff.copy(), np.diag(M_diag_ff), T_expand, warnings)

    Kaa = K_ff[np.ix_(active, active)]
    Kab = K_ff[np.ix_(active, massless)]
    Kba = K_ff[np.ix_(massless, active)]
    Kbb = K_ff[np.ix_(massless, massless)]

    try:
        solve_Kbb_Kba = np.linalg.solve(Kbb, Kba)
    except np.linalg.LinAlgError:
        Kbb_reg = Kbb + reg * np.eye(Kbb.shape[0])
        warnings.append(f"Kbb singular for massless DOFs {massless.tolist()}; regularized with reg={reg:.1e}.")
        try:
            solve_Kbb_Kba = np.linalg.solve(Kbb_reg, Kba)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Unable to condense massless DOFs; singular Kbb for local DOFs {massless.tolist()}.") from exc

    K_eff = Kaa - Kab @ solve_Kbb_Kba
    _validate_symmetric("K_eff", K_eff, tol=1e-8)
    M_eff = np.diag(M_diag_ff[active])

    T_expand = np.zeros((K_ff.shape[0], active.size), dtype=float)
    T_expand[active, np.arange(active.size)] = 1.0
    T_expand[massless, :] = -solve_Kbb_Kba

    return CondensedSystem(free_dofs, active, massless, K_eff, M_eff, T_expand, warnings)


def solve_eigenpairs(K, M, n_modes=6, *, tol=1e-10, backend="auto") -> tuple[np.ndarray, np.ndarray]:
    """Return omega_rad_s and mass-normalized modes."""
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    _validate_symmetric("K", K)
    _validate_symmetric("M", M)
    if K.shape != M.shape:
        raise ValueError(f"K and M shapes must match, got {K.shape} and {M.shape}.")
    n = K.shape[0]
    if n == 0:
        raise ValueError("Empty system.")

    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    n_modes = min(int(n_modes), n)

    m_diag = np.diag(M)
    if np.any(m_diag <= 0.0):
        raise ValueError("Mass matrix diagonal entries must be strictly positive in condensed system.")

    use_sparse = backend == "sparse" or (backend == "auto" and n > 3 and n_modes < n - 1)
    if backend not in {"auto", "dense", "sparse"}:
        raise ValueError("backend must be one of {'auto','dense','sparse'}.")

    if use_sparse:
        vals, vecs = eigsh(csc_matrix(K), k=n_modes, M=csc_matrix(M), sigma=0.0, which="LM", tol=tol)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
    else:
        vals, vecs = eigh(K, M)
        vals = vals[:n_modes]
        vecs = vecs[:, :n_modes]

    if np.any(vals < -tol):
        raise ValueError(f"Negative eigenvalues detected: min(lambda)={vals.min():.3e}")

    vals = np.clip(vals, 0.0, None)
    omega = np.sqrt(vals)

    for i in range(vecs.shape[1]):
        m_norm = float(vecs[:, i].T @ M @ vecs[:, i])
        if m_norm <= 0.0:
            raise ValueError(f"Non-positive modal mass norm for mode {i}.")
        vecs[:, i] /= np.sqrt(m_norm)

    return omega, vecs


def lanczos_own(*args, **kwargs):
    raise NotImplementedError("Educational Lanczos backend is intentionally not implemented in Phase 2.")
