def test_b1_afi_fit_raw_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.b1 import B1Afi

    model = B1Afi(nom_fa_deg=60.0, tr1_s=0.02, tr2_s=0.1)

    # Construct signals consistent with a known B1 using AFI equation
    b1_true = 1.1
    n = model.tr2_s / model.tr1_s
    afi_deg = model.nom_fa_deg * b1_true
    cos_arg = np.cos(np.deg2rad(afi_deg))
    r = (n * cos_arg + 1.0) / (n + cos_arg)

    s1 = 1000.0
    s2 = r * s1

    fitted = model.fit_raw([s1, s2])
    assert abs(fitted["b1_raw"] - b1_true) < 1e-6
    assert fitted["spurious"] == 0.0

    img = np.stack([[s1, s2], [s1, s2]], axis=0).reshape(2, 1, 2)
    out = model.fit_image(img)
    assert out["b1_raw"].shape == img.shape[:-1]
    assert out["spurious"].shape == img.shape[:-1]
