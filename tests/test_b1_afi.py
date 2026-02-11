def test_b1_afi_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.b1 import B1AFI

    model = B1AFI(nom_fa_deg=60.0, tr1_ms=20.0, tr2_ms=100.0)

    # Construct signals consistent with a known B1 using AFI equation
    b1_true = 1.1
    n = model.tr2_ms / model.tr1_ms
    afi_deg = model.nom_fa_deg * b1_true
    cos_arg = np.cos(np.deg2rad(afi_deg))
    r = (n * cos_arg + 1.0) / (n + cos_arg)

    s1 = 1000.0
    s2 = r * s1

    fitted = model.fit([s1, s2])
    assert abs(fitted["params"]["b1_raw"] - b1_true) < 1e-6
    assert fitted["diagnostics"]["spurious"] == 0.0

    img = np.stack([[s1, s2], [s1, s2]], axis=0).reshape(2, 1, 2)
    out = model.fit_image(img)
    assert out["params"]["b1_raw"].shape == img.shape[:-1]
    assert out["diagnostics"]["spurious"].shape == img.shape[:-1]


def test_b1_afi_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.b1 import B1AFI

    model = B1AFI(nom_fa_deg=60.0, tr1_ms=20.0, tr2_ms=100.0)
    signal = np.array([1000.0, 900.0], dtype=float)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))
