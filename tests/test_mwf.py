def test_mwf_two_point_basis_recovers_exact_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import MultiComponentT2

    te_ms = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0], dtype=float)
    basis = np.array([20.0, 80.0], dtype=float)
    model = MultiComponentT2(te_ms=te_ms, t2_basis_ms=basis)

    m0 = 1000.0
    mwf_true = 0.15
    signal = m0 * (mwf_true * np.exp(-te_ms / 20.0) + (1.0 - mwf_true) * np.exp(-te_ms / 80.0))

    out = model.fit(
        signal,
        regularization_alpha=0.0,
        lower_cutoff_mw_ms=None,
        cutoff_ms=40.0,
        upper_cutoff_iew_ms=200.0,
    )
    assert abs(out["mwf"] - mwf_true) < 1e-12
    assert abs(out["t2mw_ms"] - 20.0) < 1e-12
    assert abs(out["t2iew_ms"] - 80.0) < 1e-12
    assert out["resid_l2"] < 1e-9

    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)
    out_img = model.fit_image(img)
    assert out_img["mwf"].shape == img.shape[:-1]
    assert out_img["t2mw_ms"].shape == img.shape[:-1]


def test_mwf_default_basis_is_reasonable_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import MultiComponentT2

    te_ms = np.arange(10.0, 330.0, 10.0, dtype=float)
    model = MultiComponentT2(te_ms=te_ms)  # default basis (log-spaced)

    m0 = 1000.0
    mwf_true = 0.15
    signal = m0 * (mwf_true * np.exp(-te_ms / 20.0) + (1.0 - mwf_true) * np.exp(-te_ms / 80.0))

    out = model.fit(
        signal,
        regularization_alpha=1e-6,
        lower_cutoff_mw_ms=None,
        cutoff_ms=40.0,
        upper_cutoff_iew_ms=200.0,
    )
    assert abs(out["mwf"] - mwf_true) < 0.02
    assert abs(out["t2mw_ms"] - 20.0) < 10.0
    assert abs(out["t2iew_ms"] - 80.0) < 20.0
