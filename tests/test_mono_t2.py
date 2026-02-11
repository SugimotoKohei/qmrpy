def test_mono_t2_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2Mono

    te_ms = np.array([10.0, 20.0, 40.0, 80.0], dtype=float)
    model = T2Mono(te_ms=te_ms)
    signal = model.forward(m0=1000.0, t2_ms=80.0)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))


def test_mono_t2_fit_image_offset_map_behavior():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2Mono

    te_ms = np.array([10.0, 20.0, 40.0, 80.0], dtype=float)
    model = T2Mono(te_ms=te_ms)
    signal = model.forward(m0=1000.0, t2_ms=80.0)
    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)

    out = model.fit_image(img, offset_term=True)
    assert "offset" in out["params"]
    assert out["params"]["offset"].shape == img.shape[:-1]

    out_no = model.fit_image(img, offset_term=False)
    assert "offset" not in out_no["params"]
