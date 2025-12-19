"""Dynamic integrators.

This package provides a small set of time integration schemes and a dispatcher
that normalizes call signatures across integrators.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

from .hht_alpha import hht_alpha_newton
from .newmark import newmark_beta_newton
from .explicit import explicit_verlet


_INTEGRATORS = {
    "hht": hht_alpha_newton,
    "newmark": newmark_beta_newton,
    "explicit": explicit_verlet,
}


def solve_dynamic(integrator: str = "hht", /, **kwargs: Any) -> Dict:
    """Dispatch to the selected integrator.

    Parameters
    ----------
    integrator:
        One of: 'hht', 'newmark', 'explicit'.
    kwargs:
        Arguments accepted by the integrator function. Extra kwargs are ignored
        (this is intentional to allow a common call-site).
    """
    key = str(integrator).strip().lower()
    if key not in _INTEGRATORS:
        raise ValueError(f"Unknown integrator '{integrator}'. Choose one of: {sorted(_INTEGRATORS)}")
    fn: Callable[..., Dict] = _INTEGRATORS[key]
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


__all__ = [
    "hht_alpha_newton",
    "newmark_beta_newton",
    "explicit_verlet",
    "solve_dynamic",
]
