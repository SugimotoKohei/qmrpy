def test_despot1_hifi_recovers_t1_and_b1():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T1DESPOT1HIFI

    model = T1DESPOT1HIFI(flip_angle_deg=np.array([3.0, 10.0, 18.0]), tr_ms=18.0)

    signal = model.forward(m0=1000.0, t1_ms=1100.0, b1=0.9)
    out = model.fit(signal, estimate_b1=True)

    assert abs(out["params"]["t1_ms"] - 1100.0) / 1100.0 < 0.05
    assert abs(out["params"]["b1"] - 0.9) < 0.1


def test_mp2rage_fit_nls_and_lut():
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T1MP2RAGE

    model = T1MP2RAGE(ti1_ms=700.0, ti2_ms=2500.0, alpha1_deg=4.0, alpha2_deg=5.0)
    signal = model.forward(m0=900.0, t1_ms=1200.0, b1=1.05)

    out_nls = model.fit(signal, method="nls", estimate_b1=True)
    out_lut = model.fit(signal, method="lut", estimate_b1=True)

    assert abs(out_nls["params"]["t1_ms"] - 1200.0) / 1200.0 < 0.1
    assert abs(out_lut["params"]["t1_ms"] - 1200.0) / 1200.0 < 0.15


def test_mp2rage_fit_image_shape():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T1MP2RAGE

    model = T1MP2RAGE(ti1_ms=700.0, ti2_ms=2500.0, alpha1_deg=4.0, alpha2_deg=5.0)
    signal = model.forward(m0=900.0, t1_ms=1200.0, b1=1.0)
    img = np.stack([signal, signal], axis=0).reshape(2, 1, 2)

    out = model.fit_image(img)
    assert out["params"]["t1_ms"].shape == (2, 1)
    assert out["params"]["b1"].shape == (2, 1)


def test_mp2rage_rejects_empty_t1_grid():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T1MP2RAGE

    model = T1MP2RAGE(ti1_ms=700.0, ti2_ms=2500.0, alpha1_deg=4.0, alpha2_deg=5.0)
    signal = model.forward(m0=900.0, t1_ms=1200.0, b1=1.0)

    with pytest.raises(ValueError, match="t1_grid_ms must not be empty"):
        model.fit(signal, method="nls", estimate_b1=True, t1_grid_ms=np.array([], dtype=np.float64))
