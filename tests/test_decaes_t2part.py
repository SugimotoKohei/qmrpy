import numpy as np

from qmrpy.models.t2.decaes_t2part import DecaesT2Part


def test_t2part_basic_windows() -> None:
    n = 40
    part = DecaesT2Part(
        n_t2=n,
        t2_range_ms=(10.0, 2000.0),
        spwin_ms=(10.0, 25.0),
        mpwin_ms=(25.0, 200.0),
        sigmoid_ms=None,
    )

    t2 = part.t2_times_ms()
    dist = np.zeros(n)
    dist[(t2 >= 12.0) & (t2 <= 20.0)] = 1.0  # short pool
    dist[(t2 >= 50.0) & (t2 <= 120.0)] = 3.0  # medium pool

    out = part.fit(dist)
    assert 0.0 <= out["sfr"] <= 1.0
    assert 0.0 <= out["mfr"] <= 1.0
    assert out["mfr"] > out["sfr"]
    assert np.isfinite(out["sgm"])
    assert np.isfinite(out["mgm"])


def test_t2part_sigmoid_runs() -> None:
    part = DecaesT2Part(
        n_t2=40,
        t2_range_ms=(10.0, 2000.0),
        spwin_ms=(10.0, 25.0),
        mpwin_ms=(25.0, 200.0),
        sigmoid_ms=5.0,
    )

    dist = np.ones(40)
    out = part.fit(dist)
    assert 0.0 <= out["sfr"] <= 1.0
