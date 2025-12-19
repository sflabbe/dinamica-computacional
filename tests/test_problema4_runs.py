from __future__ import annotations

import numpy as np

from problems import problema4_portico


def test_problema4_runs_short():
    peak_drifts, amps_used, last, model, meta = problema4_portico.run_incremental_amplitudes(
        amps_g=np.array([0.1]),
        t_end=2.0,
        base_dt=0.01,
        dt_min=0.0025,
    )
    assert amps_used.size == 1
    assert last is not None
    assert peak_drifts
