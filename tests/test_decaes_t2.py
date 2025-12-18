def test_epg_backends_are_close_on_grid_point():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2.decaes_t2 import epg_decay_curve

    y_epgpy = epg_decay_curve(
        etl=16,
        alpha_deg=180.0,
        te_s=0.010,
        t2_s=0.080,
        t1_s=1.0,
        beta_deg=180.0,
        backend="epgpy",
    )
    y_decaes = epg_decay_curve(
        etl=16,
        alpha_deg=180.0,
        te_s=0.010,
        t2_s=0.080,
        t1_s=1.0,
        beta_deg=180.0,
        backend="decaes",
    )

    assert y_epgpy.shape == (16,)
    assert y_decaes.shape == (16,)
    assert np.all(np.isfinite(y_epgpy))
    assert np.all(np.isfinite(y_decaes))
    # Vendored epgpy backend is expected to match DECAES backend very closely.
    assert np.max(np.abs(y_epgpy - y_decaes)) < 1e-10


def test_epg_decay_is_close_to_exponential_for_ideal_refocusing():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2.decaes_t2 import epg_decay_curve

    etl = 16
    te = 0.010
    t2 = 0.080
    t1 = 1e9  # effectively infinite

    y = epg_decay_curve(
        etl=etl,
        alpha_deg=180.0,
        te_s=te,
        t2_s=t2,
        t1_s=t1,
        beta_deg=180.0,
        backend="epgpy",
    )

    expected = np.exp(-(np.arange(1, etl + 1) * te) / t2)
    # normalize both to first echo (shape-only comparison)
    y = y / y[0]
    expected = expected / expected[0]
    assert np.allclose(y, expected, rtol=1e-3, atol=1e-6)


def test_decaes_t2_map_fit_runs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2.decaes_t2 import DecaesT2Map

    m = DecaesT2Map(
        n_te=16,
        te_s=0.010,
        n_t2=30,
        t2_range_s=(0.010, 2.0),
        set_flip_angle_deg=180.0,
        epg_backend="epgpy",
        reg="gcv",
    )

    # synthetic signal: single exponential with T2=80ms
    t2 = 0.080
    te = m.echotimes_s()
    sig = np.exp(-te / t2)
    out = m.fit(sig)
    assert "distribution" in out
    assert out["distribution"].shape == (m.n_t2,)

