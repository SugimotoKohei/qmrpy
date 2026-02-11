def test_epg_t2_fit_recovers_t2():
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2EPG

    model = T2EPG(n_te=32, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0, beta_deg=180.0)
    signal = model.forward(m0=1000.0, t2_ms=80.0)

    fitted = model.fit(signal, t2_init_ms=60.0)
    assert abs(fitted["params"]["t2_ms"] - 80.0) / 80.0 < 1e-4


def test_epg_t2_fit_image_offset_map_behavior():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2EPG

    model = T2EPG(n_te=16, te_ms=10.0, t1_ms=1000.0)
    signal = model.forward(m0=500.0, t2_ms=60.0)
    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)

    out = model.fit_image(img, offset_term=True)
    assert "offset" in out["params"]
    assert out["params"]["offset"].shape == img.shape[:-1]

    out_no = model.fit_image(img, offset_term=False)
    assert "offset" not in out_no["params"]


def test_epg_t2_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2EPG

    model = T2EPG(n_te=8, te_ms=10.0, t1_ms=1000.0)
    signal = model.forward(m0=500.0, t2_ms=60.0)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))


def test_epg_t2_fit_image_with_b1_map():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2EPG

    model = T2EPG(n_te=16, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)
    t2_true = 80.0
    signal_lo = model.forward(m0=1000.0, t2_ms=t2_true, b1=0.8)
    signal_hi = model.forward(m0=1000.0, t2_ms=t2_true, b1=1.0)
    img = np.stack([signal_lo, signal_hi], axis=0).reshape(2, 1, -1)
    b1_map = np.array([[0.8], [1.0]], dtype=float)

    out = model.fit_image(img, b1_map=b1_map)
    assert abs(out["params"]["t2_ms"][0, 0] - t2_true) / t2_true < 1e-3
    assert abs(out["params"]["t2_ms"][1, 0] - t2_true) / t2_true < 1e-3
