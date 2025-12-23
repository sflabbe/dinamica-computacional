"""Default SHM calibration helpers (v9).

This module centralizes the recommended parameter defaults for the Smooth
Hysteretic Model (SHM) hinges used in this repo.

The defaults are designed to:
  * keep beam hinges elastic under gravity preload (reasonable utilization),
  * avoid excessive softening from a dimensional mismatch,
  * produce stable implicit Newmark/HHT-α time integration in IDA.

Notes
-----
The SHM implementation in this repository follows the *moment-based* evolution
form (Sivaselvan & Reinhorn, 2000 style):

    M(θ) = α K0 θ + M*
    dM*/dθ = (1-α) K0 [1 - |M*/My|^N]

where M* has units of moment. This avoids the need for very large Bouc–Wen A
factors to match the elastic stiffness.

The axial interaction (De la Llera & Vásquez style) is applied as a reduction
of the reference yield moment under **compression**:

    My(N) = My0 * (1 - N_comp/Ncr)^η

with a minimum clamp.

This helper simply returns recommended defaults and a convenience My_eff.
"""

from __future__ import annotations

from typing import Dict


def calibrate_shm(
    My0: float,
    K0: float,
    N_comp: float = 0.0,
    N_cr: float | None = None,
    *,
    eta: float = 1.0,
    My_axial_min_frac: float = 0.6,
) -> Dict[str, float]:
    """Return v9 SHM parameter defaults and the axial-adjusted My.

    Parameters
    ----------
    My0 : float
        Reference yield moment at zero axial force.
    K0 : float
        Initial elastic rotational stiffness.
    N_comp : float
        Compression-positive axial force.
    N_cr : float or None
        Compression-positive axial capacity used in My(N). If None or <=0,
        axial interaction is disabled.
    eta : float
        Exponent controlling sensitivity of My(N).
    My_axial_min_frac : float
        Minimum allowed fraction of My0 (to avoid vanishing capacity).

    Returns
    -------
    dict
        Keys: alpha, Nshape, a, b1, b2, eta, Rs, lam, My_eff
    """
    My0 = float(My0)
    _ = float(K0)  # kept for signature symmetry; not used here

    # Defaults (v9)
    alpha = 0.03
    Nshape = 10.0
    a = 200.0
    b1 = 0.05
    b2 = 0.15
    Rs, lam = 0.0, 0.0

    My_eff = My0
    if N_cr is not None and float(N_cr) > 0.0:
        Nc = max(0.0, float(N_comp))
        x = max(0.0, min(1.0, 1.0 - Nc / float(N_cr)))
        My_eff = My0 * (x ** float(eta))
        My_eff = max(float(My_axial_min_frac) * My0, My_eff)

    return {
        "alpha": float(alpha),
        "Nshape": float(Nshape),
        "a": float(a),
        "b1": float(b1),
        "b2": float(b2),
        "eta": float(eta),
        "Rs": float(Rs),
        "lam": float(lam),
        "My_eff": float(My_eff),
    }
