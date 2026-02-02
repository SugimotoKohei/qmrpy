def test_vfa_t1_forward_and_fit_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VFAT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    model = VFAT1(flip_angle_deg=flip_angle_deg, tr_ms=15.0, b1=1.0)

    m0_true = 2000.0
    t1_true_ms = 900.0
    signal = model.forward(m0=m0_true, t1_ms=t1_true_ms)

    fitted = model.fit(signal)
    assert abs(fitted["m0"] - m0_true) / m0_true < 1e-6
    assert abs(fitted["t1_ms"] - t1_true_ms) / t1_true_ms < 1e-6


def test_vfa_t1_fit_respects_b1():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VFAT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    tr_ms = 15.0
    b1 = 0.9
    model = VFAT1(flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, b1=b1)

    m0_true = 1500.0
    t1_true_ms = 1100.0
    signal = model.forward(m0=m0_true, t1_ms=t1_true_ms)

    fitted = model.fit(signal)
    assert abs(fitted["t1_ms"] - t1_true_ms) / t1_true_ms < 1e-6


def test_vfa_t1_robust_fit_reduces_outlier_impact():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VFAT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    model = VFAT1(flip_angle_deg=flip_angle_deg, tr_ms=15.0, b1=1.0)

    m0_true = 2000.0
    t1_true_ms = 900.0
    signal = model.forward(m0=m0_true, t1_ms=t1_true_ms).copy()
    signal[1] = signal[1] * 0.2  # deterministic outlier (downward spike)

    nonrobust = model.fit(signal, robust=False)
    robust = model.fit(signal, robust=True)

    err_nonrobust = abs(nonrobust["t1_ms"] - t1_true_ms)
    err_robust = abs(robust["t1_ms"] - t1_true_ms)
    assert err_robust < err_nonrobust


def test_vfa_t1_outlier_rejection_recovers_from_upward_spike():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VFAT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    model = VFAT1(flip_angle_deg=flip_angle_deg, tr_ms=15.0, b1=1.0)

    m0_true = 2000.0
    t1_true_ms = 900.0
    signal = model.forward(m0=m0_true, t1_ms=t1_true_ms).copy()
    signal[1] = signal[1] * 5.0  # upward spike

    no_reject = model.fit(signal, outlier_reject=False)
    reject = model.fit(signal, outlier_reject=True)

    assert abs(reject["t1_ms"] - t1_true_ms) < abs(no_reject["t1_ms"] - t1_true_ms)
    assert reject["n_points"] < no_reject["n_points"]


def test_vfa_t1_fit_image_shapes():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VFAT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    model = VFAT1(flip_angle_deg=flip_angle_deg, tr_ms=15.0, b1=1.0)

    signal = model.forward(m0=2000.0, t1_ms=900.0)
    img = np.stack([signal, signal], axis=0).reshape(2, 1, -1)
    out = model.fit_image(img)

    assert out["m0"].shape == img.shape[:-1]
    assert out["t1_ms"].shape == img.shape[:-1]
    assert out["n_points"].shape == img.shape[:-1]


def test_vfa_t1_fit_image_rejects_mask_for_1d():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VFAT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    model = VFAT1(flip_angle_deg=flip_angle_deg, tr_ms=15.0, b1=1.0)
    signal = model.forward(m0=2000.0, t1_ms=900.0)

    with pytest.raises(ValueError, match="mask must be None for 1D data"):
        model.fit_image(signal, mask=np.array([1], dtype=bool))
