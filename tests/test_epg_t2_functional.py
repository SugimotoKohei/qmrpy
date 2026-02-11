def test_epg_t2_functional_roundtrip():
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy import fit_t2_epg, simulate_t2_epg

    signal = simulate_t2_epg(
        m0=1.0,
        t2_ms=80.0,
        n_te=32,
        te_ms=10.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
    )
    fit = fit_t2_epg(signal, n_te=32, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)
    assert abs(fit["params"]["t2_ms"] - 80.0) / 80.0 < 1e-3


def test_epg_t2_functional_b1():
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy import fit_t2_epg, simulate_t2_epg

    signal = simulate_t2_epg(
        m0=1.0,
        t2_ms=90.0,
        n_te=24,
        te_ms=12.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
        b1=0.9,
    )
    fit = fit_t2_epg(
        signal,
        n_te=24,
        te_ms=12.0,
        t1_ms=1000.0,
        alpha_deg=180.0,
        b1=0.9,
    )
    assert abs(fit["params"]["t2_ms"] - 90.0) / 90.0 < 1e-3
