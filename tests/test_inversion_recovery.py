def test_inversion_recovery_forward_matches_expected():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import InversionRecovery

    ti = np.array([0.0, 100.0, 200.0], dtype=float)
    model = InversionRecovery(ti_ms=ti)
    out = model.forward(t1_ms=100.0, ra=2.0, rb=-1.0, magnitude=False)
    expected = 2.0 + (-1.0) * np.exp(-ti / 100.0)
    assert np.allclose(out, expected, rtol=0, atol=1e-12)


def test_inversion_recovery_fit_complex_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t1 import InversionRecovery

    ti = np.array([350.0, 500.0, 650.0, 800.0, 950.0, 1100.0, 1250.0, 1400.0, 1700.0], dtype=float)
    model = InversionRecovery(ti_ms=ti)

    t1_true = 900.0
    ra_true = 500.0
    rb_true = -1000.0
    signal = model.forward(t1_ms=t1_true, ra=ra_true, rb=rb_true, magnitude=False)

    fitted = model.fit(signal, method="complex")
    assert abs(fitted["t1_ms"] - t1_true) / t1_true < 1e-8
    assert abs(fitted["ra"] - ra_true) / max(1.0, abs(ra_true)) < 1e-8
    assert abs(fitted["rb"] - rb_true) / max(1.0, abs(rb_true)) < 1e-8


def test_inversion_recovery_fit_magnitude_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t1 import InversionRecovery

    ti = np.array([350.0, 500.0, 650.0, 800.0, 950.0, 1100.0, 1250.0, 1400.0, 1700.0], dtype=float)
    model = InversionRecovery(ti_ms=ti)

    t1_true = 900.0
    ra_true = 500.0
    rb_true = -1000.0
    signal_mag = model.forward(t1_ms=t1_true, ra=ra_true, rb=rb_true, magnitude=True)

    fitted = model.fit(signal_mag, method="magnitude")
    assert abs(fitted["t1_ms"] - t1_true) / t1_true < 1e-6
    assert 0 <= int(fitted["idx"]) <= ti.size

    img = np.stack([signal_mag, signal_mag], axis=0).reshape(2, 1, -1)
    out = model.fit_image(img, method="magnitude")
    assert out["t1_ms"].shape == img.shape[:-1]
    assert out["idx"].shape == img.shape[:-1]

