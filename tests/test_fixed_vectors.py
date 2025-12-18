def test_fixed_vectors_mono_t2_forward_matches_expected():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2 import MonoT2

    te = np.array([0.0, 10.0, 20.0, 40.0], dtype=float)
    model = MonoT2(te=te)
    out = model.forward(m0=2.0, t2=10.0)
    expected = np.array([2.0, 2.0 * np.exp(-1.0), 2.0 * np.exp(-2.0), 2.0 * np.exp(-4.0)], dtype=float)
    assert np.allclose(out, expected, rtol=0, atol=1e-12)


def test_fixed_vectors_vfa_t1_forward_matches_expected():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VfaT1

    fa = np.array([10.0], dtype=float)
    model = VfaT1(flip_angle_deg=fa, tr_s=0.015, b1=1.0)
    out = model.forward(m0=1000.0, t1_s=1.0)

    alpha = np.deg2rad(10.0)
    e = np.exp(-0.015 / 1.0)
    expected = 1000.0 * np.sin(alpha) * (1 - e) / (1 - e * np.cos(alpha))
    assert np.allclose(out, np.array([expected], dtype=float), rtol=0, atol=1e-12)


def test_fixed_vectors_rician_noise_is_deterministic():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.sim.noise import add_rician_noise

    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    s = np.array([0.0, 1.0, 2.0], dtype=float)
    y1 = add_rician_noise(s, sigma=0.5, rng=rng1)
    y2 = add_rician_noise(s, sigma=0.5, rng=rng2)
    assert np.allclose(y1, y2, rtol=0, atol=0)

