def test_import():
    import qmrpy  # noqa: F401


def test_mono_t2_forward_and_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import MonoT2

    te_ms = np.array([10.0, 20.0, 40.0, 80.0, 160.0], dtype=float)
    model = MonoT2(te=te_ms)

    m0_true = 1000.0
    t2_true = 75.0
    signal = model.forward(m0=m0_true, t2=t2_true)

    fitted = model.fit(signal)
    # 現状fitはqMRLab寄せの正規化を行うため、m0はスケール一致しない（T2が主目的）
    assert abs(fitted["t2"] - t2_true) / t2_true < 1e-6

    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)
    out_img = model.fit_image(img)
    assert out_img["t2"].shape == img.shape[:-1]
    assert abs(out_img["t2"][0, 0] - t2_true) / t2_true < 1e-6


def test_mono_t2_linear_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2 import MonoT2

    te_ms = np.array([10.0, 20.0, 40.0, 80.0, 160.0], dtype=float)
    model = MonoT2(te=te_ms)

    m0_true = 1000.0
    t2_true = 75.0
    signal = model.forward(m0=m0_true, t2=t2_true)

    fitted = model.fit(signal, fit_type="linear")
    assert abs(fitted["t2"] - t2_true) / t2_true < 1e-10
