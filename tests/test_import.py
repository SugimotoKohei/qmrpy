def test_import():
    import qmrpy  # noqa: F401


def test_mono_t2_forward_and_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import MonoT2

    te_ms = np.array([10.0, 20.0, 40.0, 80.0, 160.0], dtype=float)
    model = MonoT2(te_ms=te_ms)

    m0_true = 1000.0
    t2_true_ms = 75.0
    signal = model.forward(m0=m0_true, t2_ms=t2_true_ms)

    fitted = model.fit(signal)
    # 現状fitはqMRLab寄せの正規化を行うため、m0はスケール一致しない（T2が主目的）
    assert abs(fitted["t2_ms"] - t2_true_ms) / t2_true_ms < 1e-6

    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)
    out_img = model.fit_image(img)
    assert out_img["t2_ms"].shape == img.shape[:-1]
    assert abs(out_img["t2_ms"][0, 0] - t2_true_ms) / t2_true_ms < 1e-6


def test_mono_t2_linear_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2 import MonoT2

    te_ms = np.array([10.0, 20.0, 40.0, 80.0, 160.0], dtype=float)
    model = MonoT2(te_ms=te_ms)

    m0_true = 1000.0
    t2_true_ms = 75.0
    signal = model.forward(m0=m0_true, t2_ms=t2_true_ms)

    fitted = model.fit(signal, fit_type="linear")
    assert abs(fitted["t2_ms"] - t2_true_ms) / t2_true_ms < 1e-10
