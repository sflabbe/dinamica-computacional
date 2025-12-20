"""Optional Numba helpers.

This project works without Numba. If Numba is installed, we JIT-compile a few
hot numeric kernels (projection in return mapping; fiber-section integration).

Disable JIT explicitly by setting:
    DC_USE_NUMBA=0
"""

from __future__ import annotations

import os
from typing import Any, Callable

_USE = os.getenv("DC_USE_NUMBA", "1").strip().lower() not in {"0", "false", "off", "no"}

try:
    if _USE:
        from numba import njit  # type: ignore
        NUMBA_AVAILABLE = True
    else:
        raise ImportError("DC_USE_NUMBA disabled")
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any) -> Callable:
        """Fallback decorator if numba is unavailable."""

        def deco(fn: Callable) -> Callable:
            return fn

        return deco

USE_NUMBA = bool(NUMBA_AVAILABLE)

__all__ = ["njit", "NUMBA_AVAILABLE", "USE_NUMBA"]
