import numpy as np

from qmrpy.models.t2.decaes_t2part import DecaesT2Part


def test_t2part_basic_windows() -> None:
    n = 40
    part = DecaesT2Part(
        n_t2=n,
        t2_range_s=(10e-3, 2.0),
        spwin_s=(10e-3, 25e-3),
        mpwin_s=(25e-3, 200e-3),
        sigmoid_s=None,
    )

    t2 = part.t2_times_s()
    dist = np.zeros(n)
    dist[(t2 >= 12e-3) & (t2 <= 20e-3)] = 1.0  # short pool
    dist[(t2 >= 50e-3) & (t2 <= 120e-3)] = 3.0  # medium pool

    out = part.fit(dist)
    assert 0.0 <= out["sfr"] <= 1.0
    assert 0.0 <= out["mfr"] <= 1.0
    assert out["mfr"] > out["sfr"]
    assert np.isfinite(out["sgm"])
    assert np.isfinite(out["mgm"])


def test_t2part_sigmoid_runs() -> None:
    part = DecaesT2Part(
        n_t2=40,
        t2_range_s=(10e-3, 2.0),
        spwin_s=(10e-3, 25e-3),
        mpwin_s=(25e-3, 200e-3),
        sigmoid_s=5e-3,
    )

    dist = np.ones(40)
    out = part.fit(dist)
    assert 0.0 <= out["sfr"] <= 1.0
