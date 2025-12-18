import numpy as np


def test_generate_4d_phantom_shapes() -> None:
    from qmrpy.sim import generate_4d_phantom

    noisy, gt, sigma = generate_4d_phantom(sx=8, sy=9, sz=4, n_vol=6, snr=10.0, seed=0)
    assert noisy.shape == (8, 9, 4, 6)
    assert gt.shape == (8, 9, 4, 6)
    assert np.isfinite(sigma)
    assert sigma > 0


def test_simulate_single_voxel_with_fit_runs() -> None:
    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import simulate_single_voxel

    model = MonoT2(te=np.array([0.01, 0.02, 0.04, 0.08], dtype=np.float64))

    out = simulate_single_voxel(
        model,
        params={"m0": 1000.0, "t2": 0.06},
        noise_model="gaussian",
        noise_snr=50.0,
        fit=True,
    )

    assert set(out.keys()) == {"signal_clean", "signal", "fit"}
    assert out["signal"].shape == (4,)
    assert out["signal_clean"].shape == (4,)
    assert "t2" in out["fit"]

