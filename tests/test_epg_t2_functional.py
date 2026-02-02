def test_epg_t2_functional_roundtrip():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy import epg_t2_fit, epg_t2_forward

    signal = epg_t2_forward(
        m0=1.0,
        t2_ms=80.0,
        n_te=32,
        te_ms=10.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
    )
    fit = epg_t2_fit(signal, n_te=32, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)
    assert abs(fit["t2_ms"] - 80.0) / 80.0 < 1e-3


def test_epg_t2_functional_b1():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy import epg_t2_fit, epg_t2_forward

    signal = epg_t2_forward(
        m0=1.0,
        t2_ms=90.0,
        n_te=24,
        te_ms=12.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
        b1=0.9,
    )
    fit = epg_t2_fit(
        signal,
        n_te=24,
        te_ms=12.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
        b1=0.9,
    )
    assert abs(fit["t2_ms"] - 90.0) / 90.0 < 1e-3
