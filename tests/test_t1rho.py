from __future__ import annotations


def test_t1rho_forward_matches_monoexponential():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import T1Rho

    tsl_ms = np.array([0.0, 10.0, 30.0, 60.0], dtype=float)
    model = T1Rho(tsl_ms=tsl_ms)

    signal = model.forward(m0=800.0, t1rho_ms=75.0)

    assert np.allclose(signal, 800.0 * np.exp(-tsl_ms / 75.0))


def test_t1rho_fit_recovers_noise_free_signal():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T1Rho

    model = T1Rho(tsl_ms=np.array([0.0, 10.0, 20.0, 40.0, 80.0], dtype=float))
    signal = model.forward(m0=1200.0, t1rho_ms=90.0)

    out = model.fit(signal)

    assert out["t1rho_ms"] == pytest.approx(90.0, rel=1e-5)
    assert out["m0"] == pytest.approx(1200.0, rel=1e-5)
    assert out["params"]["t1rho_ms"] == pytest.approx(90.0, rel=1e-5)


def test_t1rho_fit_image_supports_mask_and_parallel():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T1Rho

    model = T1Rho(tsl_ms=np.array([0.0, 10.0, 20.0, 40.0], dtype=float))
    signal_a = model.forward(m0=1000.0, t1rho_ms=60.0)
    signal_b = model.forward(m0=500.0, t1rho_ms=100.0)
    image = np.stack([signal_a, signal_b], axis=0).reshape(2, 1, -1)
    mask = np.array([[True], [False]], dtype=bool)

    out = model.fit_image(image, mask=mask, n_jobs=-1)

    assert out["t1rho_ms"].shape == image.shape[:-1]
    assert out["t1rho_ms"][0, 0] == pytest.approx(60.0, rel=1e-5)
    assert np.isnan(out["t1rho_ms"][1, 0])


def test_t1rho_rejects_invalid_inputs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import T1Rho

    with pytest.raises(ValueError, match="must not be empty"):
        T1Rho(tsl_ms=[])
    with pytest.raises(ValueError, match="finite"):
        T1Rho(tsl_ms=[0.0, np.nan])
    with pytest.raises(ValueError, match="non-negative"):
        T1Rho(tsl_ms=[0.0, -1.0])

    model = T1Rho(tsl_ms=[0.0, 10.0])
    with pytest.raises(ValueError, match="t1rho_ms"):
        model.forward(m0=1.0, t1rho_ms=0.0)
    with pytest.raises(ValueError, match="signal shape"):
        model.fit([1.0, 0.9, 0.8])
    with pytest.raises(ValueError, match="mask must be None"):
        model.fit_image([1.0, 0.9], mask=np.array([True]))


def test_t1rho_functional_wrappers():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy import fit_t1rho, simulate_t1rho

    tsl_ms = np.array([0.0, 10.0, 30.0, 60.0], dtype=float)
    signal = simulate_t1rho(m0=700.0, t1rho_ms=55.0, tsl_ms=tsl_ms)
    out = fit_t1rho(signal, tsl_ms=tsl_ms)

    assert out["t1rho_ms"] == pytest.approx(55.0, rel=1e-5)
    assert out["m0"] == pytest.approx(700.0, rel=1e-5)
