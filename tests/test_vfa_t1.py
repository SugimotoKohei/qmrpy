def test_vfa_t1_forward_and_fit_linear_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VfaT1

    flip_angle_deg = np.array([3.0, 8.0, 15.0, 25.0], dtype=float)
    model = VfaT1(flip_angle_deg=flip_angle_deg, tr_s=0.015, b1=1.0)

    m0_true = 2000.0
    t1_true_s = 0.9
    signal = model.forward(m0=m0_true, t1_s=t1_true_s)

    fitted = model.fit_linear(signal)
    assert abs(fitted["m0"] - m0_true) / m0_true < 1e-6
    assert abs(fitted["t1_s"] - t1_true_s) / t1_true_s < 1e-6

