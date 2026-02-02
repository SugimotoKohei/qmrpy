def test_epg_decay_is_close_to_exponential_for_ideal_refocusing():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2.decaes_t2 import epg_decay_curve

    etl = 16
    te = 10.0
    t2 = 80.0
    t1 = 1e12  # effectively infinite (ms)

    y = epg_decay_curve(
        etl=etl,
        alpha_deg=180.0,
        te_ms=te,
        t2_ms=t2,
        t1_ms=t1,
        beta_deg=180.0,
        backend="decaes",
    )

    expected = np.exp(-(np.arange(1, etl + 1) * te) / t2)
    # normalize both to first echo (shape-only comparison)
    y = y / y[0]
    expected = expected / expected[0]
    assert np.allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_decaes_t2_map_fit_runs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2.decaes_t2 import DECAEST2Map

    m = DECAEST2Map(
        n_te=16,
        te_ms=10.0,
        n_t2=30,
        t2_range_ms=(10.0, 2000.0),
        set_flip_angle_deg=180.0,
        reg="gcv",
    )

    # synthetic signal: single exponential with T2=80ms
    t2 = 80.0
    te = m.echotimes_ms()
    sig = np.exp(-te / t2)
    out = m.fit(sig)
    assert "distribution" in out
    assert out["distribution"].shape == (m.n_t2,)
    for key in ("echotimes_ms", "t2times_ms", "alpha_deg", "gdn", "ggm", "gva", "fnr", "snr"):
        assert key in out
