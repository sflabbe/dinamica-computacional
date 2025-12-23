"""
Numba-compiled assembly kernels for HPC optimization.

Targets Python loop overhead in Model.assemble() (#1 hotspot, 25s, 32% of runtime).

Usage:
    DC_FAST=1 enables JIT compilation (requires numba)
    DC_FAST=0 falls back to pure Python loops
"""

import os
import numpy as np

# Check if Numba is available and DC_FAST=1
_DC_FAST = os.environ.get("DC_FAST", "0") == "1"
_NUMBA_AVAILABLE = False

if _DC_FAST:
    try:
        import numba
        _NUMBA_AVAILABLE = True
    except ImportError:
        _NUMBA_AVAILABLE = False
        import warnings
        warnings.warn(
            "DC_FAST=1 but numba not installed. Falling back to pure Python. "
            "Install numba for performance: pip install numba",
            RuntimeWarning
        )


if _NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True, fastmath=True)
    def aggregate_element_stiffness(
        K_global: np.ndarray,
        R_global: np.ndarray,
        k_local: np.ndarray,
        f_local: np.ndarray,
        dofs: np.ndarray,
    ) -> None:
        """
        Aggregate local element stiffness and force into global arrays.

        This is the hot inner loop of Model.assemble() (989k calls per run).
        Numba JIT eliminates Python loop overhead.

        Args:
            K_global: Global stiffness matrix (ndof × ndof), modified in-place
            R_global: Global residual vector (ndof,), modified in-place
            k_local: Local element stiffness (6×6 for frame elements)
            f_local: Local element force (6,)
            dofs: Element DOF indices (6,)

        Returns:
            None (modifies K_global and R_global in-place)
        """
        n_dof_elem = len(dofs)
        for a in range(n_dof_elem):
            ia = dofs[a]
            R_global[ia] += f_local[a]
            for b in range(n_dof_elem):
                ib = dofs[b]
                K_global[ia, ib] += k_local[a, b]

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def aggregate_hinge_stiffness(
        K_global: np.ndarray,
        R_global: np.ndarray,
        k_hinge: np.ndarray,
        f_hinge: np.ndarray,
        dofs: np.ndarray,
    ) -> None:
        """
        Aggregate local hinge stiffness and force into global arrays.

        Similar to aggregate_element_stiffness but for hinges (variable DOF count).

        Args:
            K_global: Global stiffness matrix (ndof × ndof), modified in-place
            R_global: Global residual vector (ndof,), modified in-place
            k_hinge: Local hinge stiffness (variable size)
            f_hinge: Local hinge force (variable size)
            dofs: Hinge DOF indices (variable size)

        Returns:
            None (modifies K_global and R_global in-place)
        """
        n_dof_hinge = len(dofs)
        for a in range(n_dof_hinge):
            ia = dofs[a]
            R_global[ia] += f_hinge[a]
            for b in range(n_dof_hinge):
                ib = dofs[b]
                K_global[ia, ib] += k_hinge[a, b]

else:
    # Fallback: pure Python implementation (no JIT)
    def aggregate_element_stiffness(
        K_global: np.ndarray,
        R_global: np.ndarray,
        k_local: np.ndarray,
        f_local: np.ndarray,
        dofs: np.ndarray,
    ) -> None:
        """Pure Python fallback (no JIT). See JIT version for docs."""
        for a, ia in enumerate(dofs):
            R_global[ia] += f_local[a]
            for b, ib in enumerate(dofs):
                K_global[ia, ib] += k_local[a, b]

    def aggregate_hinge_stiffness(
        K_global: np.ndarray,
        R_global: np.ndarray,
        k_hinge: np.ndarray,
        f_hinge: np.ndarray,
        dofs: np.ndarray,
    ) -> None:
        """Pure Python fallback (no JIT). See JIT version for docs."""
        for a, ia in enumerate(dofs):
            R_global[ia] += f_hinge[a]
            for b, ib in enumerate(dofs):
                K_global[ia, ib] += k_hinge[a, b]


# Public API
__all__ = [
    "aggregate_element_stiffness",
    "aggregate_hinge_stiffness",
    "is_jit_enabled",
]


def is_jit_enabled() -> bool:
    """Check if Numba JIT is enabled and available."""
    return _NUMBA_AVAILABLE
