def test_b0_dual_echo_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B0DualEcho

    model = B0DualEcho(te1_ms=5.0, te2_ms=7.0)
    b0_true = 50.0
    phase0_true = 0.3

    phase = model.forward(b0_hz=b0_true, phase0_rad=phase0_true)
    signal = np.exp(1j * phase)

    out = model.fit(signal)
    assert abs(out["params"]["b0_hz"] - b0_true) < 1e-10


def test_b0_dual_echo_fit_image_shape():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B0DualEcho

    model = B0DualEcho(te1_ms=5.0, te2_ms=7.0)
    phase = model.forward(b0_hz=40.0, phase0_rad=0.1)
    signal = np.stack([np.exp(1j * phase), np.exp(1j * phase)], axis=0).reshape(2, 1, 2)

    out = model.fit_image(signal)
    assert out["params"]["b0_hz"].shape == (2, 1)
    assert out["params"]["phase0_rad"].shape == (2, 1)


def test_b0_multi_echo_fit_with_wrap():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B0MultiEcho

    te_ms = np.array([4.0, 8.0, 12.0, 16.0], dtype=float)
    model = B0MultiEcho(te_ms=te_ms, unwrap_phase=True)

    b0_true = 80.0
    phase0 = 0.1
    te_s = te_ms / 1000.0
    phase = phase0 + 2.0 * np.pi * b0_true * te_s
    wrapped = np.angle(np.exp(1j * phase))

    out = model.fit(wrapped)
    assert abs(out["params"]["b0_hz"] - b0_true) < 1e-8


def test_b0_multi_echo_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import B0MultiEcho

    model = B0MultiEcho(te_ms=np.array([4.0, 8.0, 12.0], dtype=float))
    signal = np.array([0.1, 0.2, 0.3], dtype=float)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))
