from __future__ import annotations


def test_mtr_forward_and_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MTR

    model = MTR()
    signal = model.forward(s0=1000.0, mtr=0.25)
    out = model.fit(signal)

    assert np.allclose(signal, [1000.0, 750.0])
    assert out["mtr"] == pytest.approx(0.25)
    assert out["mtr_percent"] == pytest.approx(25.0)
    assert out["params"]["mtr"] == pytest.approx(0.25)


def test_mtr_fit_image_supports_mask_and_parallel():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MTR

    model = MTR()
    image = np.array([[[1000.0, 800.0]], [[500.0, 250.0]]], dtype=float)
    mask = np.array([[True], [False]], dtype=bool)

    out = model.fit_image(image, mask=mask, n_jobs=-1)

    assert out["mtr"].shape == image.shape[:-1]
    assert out["mtr"][0, 0] == pytest.approx(0.2)
    assert np.isnan(out["mtr"][1, 0])


def test_mtsat_forward_and_fit_noise_free():
    import pytest

    pytest.importorskip("numpy")

    from qmrpy.models import MTsat

    model = MTsat(flip_angle_deg=6.0, tr_ms=25.0)
    signal = model.forward(m0=1200.0, t1_ms=1000.0, mtsat=0.08)
    out = model.fit(signal, m0=1200.0, t1_ms=1000.0)

    assert out["mtsat"] == pytest.approx(0.08, rel=1e-12)
    assert out["mtsat_percent"] == pytest.approx(8.0, rel=1e-12)
    assert out["params"]["mtsat"] == pytest.approx(0.08, rel=1e-12)


def test_mtsat_fit_image_supports_maps_and_parallel():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MTsat

    model = MTsat(flip_angle_deg=6.0, tr_ms=25.0)
    signal_a = model.forward(m0=1000.0, t1_ms=900.0, mtsat=0.05)
    signal_b = model.forward(m0=800.0, t1_ms=1200.0, mtsat=0.10)
    image = np.stack([signal_a, signal_b], axis=0).reshape(2, 1, 1)
    m0 = np.array([[1000.0], [800.0]], dtype=float)
    t1_ms = np.array([[900.0], [1200.0]], dtype=float)
    mask = np.array([[True], [False]], dtype=bool)

    out = model.fit_image(image, m0=m0, t1_ms=t1_ms, mask=mask, n_jobs=-1)

    assert out["mtsat"].shape == image.shape[:-1]
    assert out["mtsat"][0, 0] == pytest.approx(0.05, rel=1e-12)
    assert np.isnan(out["mtsat"][1, 0])


def test_mt_rejects_invalid_inputs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MTR, MTsat

    mtr = MTR()
    with pytest.raises(ValueError, match="shape"):
        mtr.fit([1.0])
    with pytest.raises(ValueError, match="S0"):
        mtr.fit([0.0, 0.0])
    with pytest.raises(ValueError, match="mask must be None"):
        mtr.fit_image([100.0, 80.0], mask=np.array([True]))

    with pytest.raises(ValueError, match="flip_angle_deg"):
        MTsat(flip_angle_deg=0.0, tr_ms=25.0)
    mtsat = MTsat(flip_angle_deg=6.0, tr_ms=25.0)
    with pytest.raises(ValueError, match="shape"):
        mtsat.fit([1.0, 2.0], m0=1000.0, t1_ms=1000.0)
    with pytest.raises(ValueError, match="m0 shape"):
        mtsat.fit_image(np.ones((2, 1, 1)), m0=np.ones((2,)), t1_ms=np.ones((2, 1)))


def test_mt_functional_wrappers():
    import pytest

    pytest.importorskip("numpy")

    from qmrpy import fit_mtr, fit_mtsat, simulate_mtr, simulate_mtsat

    mtr_signal = simulate_mtr(s0=900.0, mtr=0.3)
    mtr_out = fit_mtr(mtr_signal)
    assert mtr_out["mtr"] == pytest.approx(0.3)

    mtsat_signal = simulate_mtsat(
        m0=1000.0,
        t1_ms=1000.0,
        mtsat=0.06,
        flip_angle_deg=6.0,
        tr_ms=25.0,
    )
    mtsat_out = fit_mtsat(
        mtsat_signal,
        m0=1000.0,
        t1_ms=1000.0,
        flip_angle_deg=6.0,
        tr_ms=25.0,
    )
    assert mtsat_out["mtsat"] == pytest.approx(0.06, rel=1e-12)
