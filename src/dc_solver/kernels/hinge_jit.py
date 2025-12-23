"""Numba JIT kernels for SHM hinge evaluation (Bouc-Wen integration).

This module provides optimized kernels for the SHM (Smooth Hysteretic Model)
hinge evaluation, specifically the Bouc-Wen ODE integration loop.

Activation: Set DC_FAST=1 environment variable to enable JIT compilation.
            If DC_FAST=0 or unset, falls back to pure Python (identical results).

Pattern:
    - Import-time detection of Numba availability and DC_FAST setting
    - Compile-time specialization (nopython mode, cache enabled)
    - Pure-Python fallback with identical signature and outputs
"""

from __future__ import annotations

import math
import os
from typing import Tuple

# Detect DC_FAST environment variable
_DC_FAST = os.getenv("DC_FAST", "0") == "1"

# Try to import numba if DC_FAST is enabled
_NUMBA_AVAILABLE = False
if _DC_FAST:
    try:
        import numba
        _NUMBA_AVAILABLE = True
    except ImportError:
        print("Warning: DC_FAST=1 but numba is not installed. Falling back to pure Python.")


# ============================================================================
# Pure Python Implementation (always available, fallback)
# ============================================================================

def _bw_rhs_python(z: float, sign_dth: float, A_eff: float, bw_beta: float, bw_gamma: float, bw_n: float) -> float:
    """Bouc-Wen RHS evaluation (pure Python)."""
    return A_eff - bw_beta * sign_dth * abs(z) ** (bw_n - 1.0) * z - bw_gamma * abs(z) ** bw_n


def _degraded_K0_python(a: float, K0_0: float, b1: float, cK: float, K0_min_frac: float) -> float:
    """Degraded stiffness (pure Python)."""
    fac_lin = max(K0_min_frac, 1.0 - b1 * a)
    fac_exp = math.exp(-cK * a) if cK != 0.0 else 1.0
    return max(K0_min_frac * K0_0, K0_0 * fac_lin * fac_exp)


def _degraded_My_python(a: float, My0: float, b2: float, cMy: float, My_min_frac: float) -> float:
    """Degraded moment capacity (pure Python)."""
    fac_lin = max(My_min_frac, 1.0 - b2 * a)
    fac_exp = math.exp(-cMy * a) if cMy != 0.0 else 1.0
    return max(My_min_frac * My0, My0 * fac_lin * fac_exp)


def shm_bouc_wen_step_python(
    dth: float,
    th_comm: float,
    z_comm: float,
    a_comm: float,
    M_comm: float,
    # Model parameters
    K0_0: float,
    My0_abs: float,
    alpha_post: float,
    bw_A: float,
    bw_beta: float,
    bw_gamma: float,
    bw_n: float,
    b1: float,
    b2: float,
    cK: float,
    cMy: float,
    K0_min_frac: float,
    My_min_frac: float,
    Eref_dimless: float,
    pinch: float,
    theta_pinch: float,
    dth_sub_max: float,
    max_substeps: int,
) -> Tuple[float, float, float, float, float, float]:
    """
    SHM Bouc-Wen integration step (pure Python).

    Returns: (M, k_tan, th, z, a, M_residual)
    """
    # Determine number of substeps
    if dth_sub_max > 0.0:
        nsub = max(1, int(math.ceil(abs(dth) / max(dth_sub_max, 1e-12))))
    else:
        nsub = 1
    nsub = min(max(nsub, 1), max(max_substeps, 1))
    dth_sub = dth / float(nsub)

    # Initialize state
    th = th_comm
    z = z_comm
    a = a_comm
    M = M_comm

    # Auto-scale Bouc-Wen A
    if bw_A > 0.0:
        A_eff = bw_A
    else:
        A_eff = max(K0_0 / max(My0_abs, 1e-12), 1e-6)

    # Substep loop (RK4 integration)
    for _ in range(nsub):
        th_new = th + dth_sub
        sign_dth = 1.0 if dth_sub >= 0.0 else -1.0

        # RK4 for z(theta)
        k1 = _bw_rhs_python(z, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
        k2 = _bw_rhs_python(z + 0.5 * dth_sub * k1, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
        k3 = _bw_rhs_python(z + 0.5 * dth_sub * k2, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
        k4 = _bw_rhs_python(z + dth_sub * k3, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
        z_new = z + (dth_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Degraded properties
        K0 = _degraded_K0_python(a, K0_0, b1, cK, K0_min_frac)
        My = _degraded_My_python(a, My0_abs, b2, cMy, My_min_frac)

        # Pinching
        pinch_factor = 1.0
        if pinch > 0.0 and (th_new * sign_dth) < 0.0:
            pinch_factor = 1.0 - pinch * math.exp(-abs(th_new) / max(theta_pinch, 1e-12))

        M_new = alpha_post * K0 * th_new + (1.0 - alpha_post) * My * z_new * pinch_factor

        # Accumulate hysteretic work
        dW = 0.5 * (M + M_new) * dth_sub
        a += abs(dW) / max(Eref_dimless, 1e-18)

        # Update state
        th = th_new
        z = max(-1.2, min(1.2, z_new))
        M = M_new

    # Tangent stiffness
    K0 = _degraded_K0_python(a, K0_0, b1, cK, K0_min_frac)
    My = _degraded_My_python(a, My0_abs, b2, cMy, My_min_frac)
    sign_tot = 1.0 if dth >= 0.0 else -1.0
    dz_dth = _bw_rhs_python(z, sign_tot, A_eff, bw_beta, bw_gamma, bw_n)
    dz_dth = max(0.0, dz_dth)

    pinch_factor_end = 1.0
    if pinch > 0.0 and (th * sign_tot) < 0.0:
        pinch_factor_end = 1.0 - pinch * math.exp(-abs(th) / max(theta_pinch, 1e-12))

    k_tan = alpha_post * K0 + (1.0 - alpha_post) * My * dz_dth * pinch_factor_end
    k_tan = max(k_tan, max(alpha_post * K0, 1e-12))

    return M, k_tan, th, z, a, M


# ============================================================================
# Numba JIT Implementation (when DC_FAST=1 and numba available)
# ============================================================================

if _NUMBA_AVAILABLE:

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _bw_rhs_jit(z: float, sign_dth: float, A_eff: float, bw_beta: float, bw_gamma: float, bw_n: float) -> float:
        """Bouc-Wen RHS evaluation (JIT)."""
        return A_eff - bw_beta * sign_dth * abs(z) ** (bw_n - 1.0) * z - bw_gamma * abs(z) ** bw_n

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _degraded_K0_jit(a: float, K0_0: float, b1: float, cK: float, K0_min_frac: float) -> float:
        """Degraded stiffness (JIT)."""
        fac_lin = max(K0_min_frac, 1.0 - b1 * a)
        fac_exp = math.exp(-cK * a) if cK != 0.0 else 1.0
        return max(K0_min_frac * K0_0, K0_0 * fac_lin * fac_exp)

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def _degraded_My_jit(a: float, My0: float, b2: float, cMy: float, My_min_frac: float) -> float:
        """Degraded moment capacity (JIT)."""
        fac_lin = max(My_min_frac, 1.0 - b2 * a)
        fac_exp = math.exp(-cMy * a) if cMy != 0.0 else 1.0
        return max(My_min_frac * My0, My0 * fac_lin * fac_exp)

    @numba.jit(nopython=True, cache=True, fastmath=True)
    def shm_bouc_wen_step_jit(
        dth: float,
        th_comm: float,
        z_comm: float,
        a_comm: float,
        M_comm: float,
        # Model parameters
        K0_0: float,
        My0_abs: float,
        alpha_post: float,
        bw_A: float,
        bw_beta: float,
        bw_gamma: float,
        bw_n: float,
        b1: float,
        b2: float,
        cK: float,
        cMy: float,
        K0_min_frac: float,
        My_min_frac: float,
        Eref_dimless: float,
        pinch: float,
        theta_pinch: float,
        dth_sub_max: float,
        max_substeps: int,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        SHM Bouc-Wen integration step (Numba JIT).

        Returns: (M, k_tan, th, z, a, M_residual)
        """
        # Determine number of substeps
        if dth_sub_max > 0.0:
            nsub = max(1, int(math.ceil(abs(dth) / max(dth_sub_max, 1e-12))))
        else:
            nsub = 1
        nsub = min(max(nsub, 1), max(max_substeps, 1))
        dth_sub = dth / float(nsub)

        # Initialize state
        th = th_comm
        z = z_comm
        a = a_comm
        M = M_comm

        # Auto-scale Bouc-Wen A
        if bw_A > 0.0:
            A_eff = bw_A
        else:
            A_eff = max(K0_0 / max(My0_abs, 1e-12), 1e-6)

        # Substep loop (RK4 integration)
        for _ in range(nsub):
            th_new = th + dth_sub
            sign_dth = 1.0 if dth_sub >= 0.0 else -1.0

            # RK4 for z(theta)
            k1 = _bw_rhs_jit(z, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
            k2 = _bw_rhs_jit(z + 0.5 * dth_sub * k1, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
            k3 = _bw_rhs_jit(z + 0.5 * dth_sub * k2, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
            k4 = _bw_rhs_jit(z + dth_sub * k3, sign_dth, A_eff, bw_beta, bw_gamma, bw_n)
            z_new = z + (dth_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Degraded properties
            K0 = _degraded_K0_jit(a, K0_0, b1, cK, K0_min_frac)
            My = _degraded_My_jit(a, My0_abs, b2, cMy, My_min_frac)

            # Pinching
            pinch_factor = 1.0
            if pinch > 0.0 and (th_new * sign_dth) < 0.0:
                pinch_factor = 1.0 - pinch * math.exp(-abs(th_new) / max(theta_pinch, 1e-12))

            M_new = alpha_post * K0 * th_new + (1.0 - alpha_post) * My * z_new * pinch_factor

            # Accumulate hysteretic work
            dW = 0.5 * (M + M_new) * dth_sub
            a += abs(dW) / max(Eref_dimless, 1e-18)

            # Update state
            th = th_new
            z = max(-1.2, min(1.2, z_new))
            M = M_new

        # Tangent stiffness
        K0 = _degraded_K0_jit(a, K0_0, b1, cK, K0_min_frac)
        My = _degraded_My_jit(a, My0_abs, b2, cMy, My_min_frac)
        sign_tot = 1.0 if dth >= 0.0 else -1.0
        dz_dth = _bw_rhs_jit(z, sign_tot, A_eff, bw_beta, bw_gamma, bw_n)
        dz_dth = max(0.0, dz_dth)

        pinch_factor_end = 1.0
        if pinch > 0.0 and (th * sign_tot) < 0.0:
            pinch_factor_end = 1.0 - pinch * math.exp(-abs(th) / max(theta_pinch, 1e-12))

        k_tan = alpha_post * K0 + (1.0 - alpha_post) * My * dz_dth * pinch_factor_end
        k_tan = max(k_tan, max(alpha_post * K0, 1e-12))

        return M, k_tan, th, z, a, M

    # Use JIT version
    shm_bouc_wen_step = shm_bouc_wen_step_jit

else:
    # Use pure Python version
    shm_bouc_wen_step = shm_bouc_wen_step_python


# ============================================================================
# Public API
# ============================================================================

def is_jit_enabled() -> bool:
    """Return True if JIT compilation is enabled."""
    return _NUMBA_AVAILABLE


def get_backend() -> str:
    """Return current backend: 'numba' or 'python'."""
    return "numba" if _NUMBA_AVAILABLE else "python"
