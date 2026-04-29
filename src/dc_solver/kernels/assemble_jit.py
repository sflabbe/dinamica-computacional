"""Numba JIT kernels for FE assembly (scatter-add aggregation).

This module provides optimized kernels for assembling element stiffness matrices
and force vectors into the global system.

Activation: Set DC_FAST=1 environment variable to enable JIT compilation.
            If DC_FAST=0 or unset, falls back to pure Python (identical results).

Pattern:
    - Import-time detection of Numba availability and DC_FAST setting
    - Compile-time specialization (nopython mode, cache enabled)
    - Pure-Python fallback with identical signature and outputs
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

# DC_USE_NUMBA=0 explicitly disables all JIT; DC_FAST=1 or DC_USE_NUMBA=1 activates it.
_DC_USE_NUMBA_ENV = os.getenv("DC_USE_NUMBA", "").strip().lower()
_DC_FAST = os.getenv("DC_FAST", "0") == "1"

_NUMBA_DISABLED = _DC_USE_NUMBA_ENV in {"0", "false", "off", "no"}
_NUMBA_REQUESTED = _DC_FAST or (_DC_USE_NUMBA_ENV not in {"", "0", "false", "off", "no"})

# Try to import numba if enabled
_NUMBA_AVAILABLE = False
if not _NUMBA_DISABLED and _NUMBA_REQUESTED:
    try:
        import numba
        _NUMBA_AVAILABLE = True
    except ImportError:
        print("Warning: numba requested but not installed. Falling back to pure Python.")


# ============================================================================
# Pure Python Implementation (always available, fallback)
# ============================================================================

def assemble_elements_python(
    K_global: np.ndarray,
    R_global: np.ndarray,
    k_locals: np.ndarray,
    f_locals: np.ndarray,
    dof_maps: np.ndarray,
) -> None:
    """
    Assemble element stiffness matrices and force vectors into global arrays.

    Parameters
    ----------
    K_global : ndarray (ndof, ndof)
        Global stiffness matrix (modified in-place)
    R_global : ndarray (ndof,)
        Global force vector (modified in-place)
    k_locals : ndarray (n_elem, n_dof_elem, n_dof_elem)
        Element stiffness matrices
    f_locals : ndarray (n_elem, n_dof_elem)
        Element force vectors
    dof_maps : ndarray (n_elem, n_dof_elem), dtype=int
        DOF mapping for each element

    Notes
    -----
    Operates in-place on K_global and R_global (scatter-add operation).
    """
    n_elem = k_locals.shape[0]
    n_dof_elem = k_locals.shape[1]

    for e in range(n_elem):
        for a in range(n_dof_elem):
            ia = dof_maps[e, a]
            R_global[ia] += f_locals[e, a]
            for b in range(n_dof_elem):
                ib = dof_maps[e, b]
                K_global[ia, ib] += k_locals[e, a, b]


# ============================================================================
# Numba JIT Implementation (when DC_FAST=1 and numba available)
# ============================================================================

if _NUMBA_AVAILABLE:

    @numba.jit(nopython=True, cache=True, parallel=False)
    def assemble_elements_jit(
        K_global: np.ndarray,
        R_global: np.ndarray,
        k_locals: np.ndarray,
        f_locals: np.ndarray,
        dof_maps: np.ndarray,
    ) -> None:
        """
        Assemble element stiffness matrices and force vectors into global arrays (JIT).

        Parameters
        ----------
        K_global : ndarray (ndof, ndof)
            Global stiffness matrix (modified in-place)
        R_global : ndarray (ndof,)
            Global force vector (modified in-place)
        k_locals : ndarray (n_elem, n_dof_elem, n_dof_elem)
            Element stiffness matrices
        f_locals : ndarray (n_elem, n_dof_elem)
            Element force vectors
        dof_maps : ndarray (n_elem, n_dof_elem), dtype=int
            DOF mapping for each element
        """
        n_elem = k_locals.shape[0]
        n_dof_elem = k_locals.shape[1]

        for e in range(n_elem):
            for a in range(n_dof_elem):
                ia = dof_maps[e, a]
                R_global[ia] += f_locals[e, a]
                for b in range(n_dof_elem):
                    ib = dof_maps[e, b]
                    K_global[ia, ib] += k_locals[e, a, b]

    # Use JIT version
    assemble_elements = assemble_elements_jit

else:
    # Use pure Python version
    assemble_elements = assemble_elements_python


# ============================================================================
# Public API
# ============================================================================

def is_jit_enabled() -> bool:
    """Return True if JIT compilation is enabled."""
    return _NUMBA_AVAILABLE


def get_backend() -> str:
    """Return current backend: 'numba' or 'python'."""
    return "numba" if _NUMBA_AVAILABLE else "python"
