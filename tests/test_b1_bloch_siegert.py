def test_b1_bloch_siegert_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B1BlochSiegert

    model = B1BlochSiegert(k_bs_rad_per_b1sq=2.0)
    b1_true = 0.6

    phase = model.forward(b1=b1_true, phase0_rad=0.2)
    signal = np.exp(1j * phase)

    out = model.fit(signal)
    assert abs(out["params"]["b1_raw"] - b1_true) < 1e-10
    assert out["diagnostics"]["spurious"] == 0.0


def test_b1_bloch_siegert_fit_image_shape():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B1BlochSiegert

    model = B1BlochSiegert(k_bs_rad_per_b1sq=1.5)
    phase = model.forward(b1=0.9)
    signal = np.stack([np.exp(1j * phase), np.exp(1j * phase)], axis=0).reshape(2, 1, 2)

    out = model.fit_image(signal)
    assert out["params"]["b1_raw"].shape == (2, 1)
    assert out["diagnostics"]["spurious"].shape == (2, 1)


def test_b1_bloch_siegert_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B1BlochSiegert

    model = B1BlochSiegert()
    signal = np.array([0.1, -0.1], dtype=float)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))
