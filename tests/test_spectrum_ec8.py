import numpy as np
import pytest

from dc_solver.modal.response_spectrum import SpectrumEC8, spectral_combination


def test_sa_positive_finite():
    sp = SpectrumEC8(ag=2.0, soil="C")
    T = np.linspace(0.0, 4.0, 50)
    Sa = sp.Sa(T)
    assert np.all(np.isfinite(Sa))
    assert np.all(Sa > 0.0)


def test_cqc_not_implemented():
    with pytest.raises(NotImplementedError):
        spectral_combination([1.0, 2.0], method="cqc")
