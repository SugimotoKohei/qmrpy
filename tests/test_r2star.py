def test_r2star_mono_fit_recovers_t2star():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2StarMonoR2

    te_ms = np.array([4.0, 8.0, 12.0, 16.0, 20.0], dtype=float)
    model = T2StarMonoR2(te_ms=te_ms)

    signal = model.forward(s0=800.0, t2star_ms=30.0)
    out = model.fit(signal)

    assert abs(out["params"]["t2star_ms"] - 30.0) / 30.0 < 1e-4
    assert abs(out["params"]["r2star_hz"] - (1000.0 / 30.0)) / (1000.0 / 30.0) < 1e-4


def test_r2star_complex_fit_recovers_params():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2StarComplexR2

    te_ms = np.array([4.0, 8.0, 12.0, 16.0, 20.0], dtype=float)
    model = T2StarComplexR2(te_ms=te_ms)

    signal = model.forward(s0=500.0, t2star_ms=25.0, delta_f_hz=12.0, phi0_rad=0.4)
    out = model.fit(signal)

    assert abs(out["params"]["t2star_ms"] - 25.0) / 25.0 < 1e-2
    assert abs(out["params"]["delta_f_hz"] - 12.0) < 0.5


def test_r2star_estatics_fit_multicontrast():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2StarESTATICS

    te_ms = np.array([3.0, 6.0, 9.0, 12.0], dtype=float)
    model = T2StarESTATICS(te_ms=te_ms)

    t2s_true = 22.0
    decay = np.exp(-te_ms / t2s_true)
    signal = np.stack([700.0 * decay, 300.0 * decay], axis=0)

    out = model.fit(signal)
    assert abs(out["params"]["t2star_ms"] - t2s_true) / t2s_true < 1e-3


def test_r2star_mono_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import T2StarMonoR2

    model = T2StarMonoR2(te_ms=np.array([4.0, 8.0], dtype=float))
    signal = np.array([100.0, 80.0], dtype=float)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))
