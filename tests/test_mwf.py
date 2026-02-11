def test_mwf_two_point_basis_recovers_exact_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2MultiComponent

    te_ms = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0], dtype=float)
    basis = np.array([20.0, 80.0], dtype=float)
    model = T2MultiComponent(te_ms=te_ms, t2_basis_ms=basis)

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
    for key in ("weights", "t2_basis_ms", "mwf", "t2mw_ms", "t2iew_ms", "gmt2_ms"):
        assert key in out["params"]
    assert abs(out["params"]["mwf"] - mwf_true) < 1e-12
    assert abs(out["params"]["t2mw_ms"] - 20.0) < 1e-12
    assert abs(out["params"]["t2iew_ms"] - 80.0) < 1e-12
    assert out["quality"]["rmse"] < 1e-9

    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)
    out_img = model.fit_image(img, return_weights=True)
    assert out_img["params"]["mwf"].shape == img.shape[:-1]
    assert out_img["params"]["t2mw_ms"].shape == img.shape[:-1]
    assert "weights" in out_img["params"]


def test_mwf_default_basis_is_reasonable_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2MultiComponent

    te_ms = np.arange(10.0, 330.0, 10.0, dtype=float)
    model = T2MultiComponent(te_ms=te_ms)  # default basis (log-spaced)

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
    assert abs(out["params"]["mwf"] - mwf_true) < 0.02
    assert abs(out["params"]["t2mw_ms"] - 20.0) < 10.0
    assert abs(out["params"]["t2iew_ms"] - 80.0) < 20.0


def test_mwf_qmrlab_regnnls_runs_noise_free():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models.t2 import T2MultiComponent

    te_ms = np.arange(10.0, 330.0, 10.0, dtype=float)
    model = T2MultiComponent(te_ms=te_ms)

    m0 = 1000.0
    mwf_true = 0.15
    signal = m0 * (mwf_true * np.exp(-te_ms / 20.0) + (1.0 - mwf_true) * np.exp(-te_ms / 80.0))

    out = model.fit(
        signal,
        regularization_mode="qmrlab_regnnls",
        qmrlab_sigma=20.0,
        lower_cutoff_mw_ms=None,
        cutoff_ms=40.0,
        upper_cutoff_iew_ms=200.0,
    )
    assert "reg_nnls_mu" in out["diagnostics"]
    assert abs(out["params"]["mwf"] - mwf_true) < 0.05
