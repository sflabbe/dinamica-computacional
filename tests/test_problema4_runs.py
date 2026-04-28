from __future__ import annotations

import numpy as np
import pytest

from problems import problema4_portico


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_dt": 0.0}, "base_dt"),
        ({"dt_min": 0.0}, "dt_min"),
        ({"max_cutbacks": -1}, "max_cutbacks"),
    ],
)
def test_problema4_rejects_invalid_time_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        problema4_portico.run_incremental_amplitudes(amps_g=np.array([0.1]), **kwargs)


@pytest.mark.slow
def test_problema4_runs_short():
    peak_drifts, amps_used, last, model, meta = problema4_portico.run_incremental_amplitudes(
        amps_g=np.array([0.1]),
        t_end=5.0,  # Increased from 2.0 for stability
        base_dt=0.005,  # Reduced from 0.01 for better convergence
        dt_min=0.00125,  # Reduced proportionally
    )
    assert amps_used.size == 1
    assert last is not None
    assert peak_drifts
