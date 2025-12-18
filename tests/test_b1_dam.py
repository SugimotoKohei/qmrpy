def test_b1_dam_fit_raw_noise_free():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.b1 import B1Dam

    model = B1Dam(alpha_deg=60.0)
    b1_true = 1.1
    m0 = 1000.0
    signal = model.forward(m0=m0, b1=b1_true)
    fitted = model.fit_raw(signal)
    assert abs(fitted["b1_raw"] - b1_true) < 1e-12
    assert fitted["spurious"] == 0.0

