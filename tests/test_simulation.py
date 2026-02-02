import numpy as np


def test_generate_4d_phantom_shapes() -> None:
    from qmrpy.sim import generate_4d_phantom

    noisy, gt, sigma = generate_4d_phantom(sx=8, sy=9, sz=4, n_vol=6, snr=10.0, seed=0)
    assert noisy.shape == (8, 9, 4, 6)
    assert gt.shape == (8, 9, 4, 6)
    assert np.isfinite(sigma)
    assert sigma > 0


def test_generate_4d_phantom_no_noise() -> None:
    from qmrpy.sim import generate_4d_phantom

    # With very high SNR (effectively no noise)
    noisy, gt, sigma = generate_4d_phantom(sx=4, sy=4, sz=2, n_vol=3, snr=1e10, seed=0)
    # Should be nearly identical
    assert np.allclose(noisy, gt, rtol=1e-3)


def test_simulate_single_voxel_with_fit_runs() -> None:
    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import SimulationProtocol, simulate_single_voxel

    model = MonoT2(te_ms=np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float64))

    out = simulate_single_voxel(
        model,
        params={"m0": 1000.0, "t2_ms": 60.0},
        simulation_backend="analytic",
        noise_model="gaussian",
        noise_snr=50.0,
        fit=True,
    )

    assert set(out.keys()) == {"signal_clean", "signal", "fit"}
    assert out["signal"].shape == (4,)
    assert out["signal_clean"].shape == (4,)
    assert "t2_ms" in out["fit"]

    proto = SimulationProtocol(simulation_backend="analytic", noise_model="none", fit=True)
    out_proto = simulate_single_voxel(
        model,
        params={"m0": 1000.0, "t2_ms": 60.0},
        protocol=proto,
        noise_model="gaussian",
        noise_sigma=10.0,
    )
    assert "fit" in out_proto
    assert out_proto["signal"].shape == out_proto["signal_clean"].shape

    proto_model = SimulationProtocol(
        simulation_backend="analytic",
        model_protocol={"te_ms": [10.0, 20.0, 40.0, 80.0]},
        fit=True,
    )
    out_from_class = simulate_single_voxel(
        MonoT2,
        params={"m0": 1000.0, "t2_ms": 60.0},
        protocol=proto_model,
    )
    assert "fit" in out_from_class


def test_simulate_single_voxel_no_noise() -> None:
    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import simulate_single_voxel

    model = MonoT2(te_ms=np.array([10.0, 20.0, 40.0], dtype=np.float64))

    out = simulate_single_voxel(
        model,
        params={"m0": 500.0, "t2_ms": 40.0},
        simulation_backend="analytic",
        noise_model="none",
        fit=False,
    )

    assert np.allclose(out["signal"], out["signal_clean"])
    assert "fit" not in out


def test_simulate_single_voxel_rician_noise() -> None:
    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import simulate_single_voxel

    model = MonoT2(te_ms=np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float64))

    out = simulate_single_voxel(
        model,
        params={"m0": 1000.0, "t2_ms": 60.0},
        simulation_backend="analytic",
        noise_model="rician",
        noise_sigma=20.0,
        rng=np.random.default_rng(42),
        fit=False,
    )

    assert out["signal"].shape == (4,)
    # Rician noise should make signal different from clean
    assert not np.allclose(out["signal"], out["signal_clean"])


def test_simulate_single_voxel_invalid_backend() -> None:
    import pytest

    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import simulate_single_voxel

    model = MonoT2(te_ms=np.array([10.0, 20.0], dtype=np.float64))

    with pytest.raises(ValueError, match="unknown simulation_backend"):
        simulate_single_voxel(
            model,
            params={"m0": 100.0, "t2_ms": 50.0},
            simulation_backend="invalid_backend",
        )


def test_simulate_single_voxel_invalid_noise_model() -> None:
    import pytest

    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import simulate_single_voxel

    model = MonoT2(te_ms=np.array([10.0, 20.0], dtype=np.float64))

    with pytest.raises(ValueError, match="unknown noise_model"):
        simulate_single_voxel(
            model,
            params={"m0": 100.0, "t2_ms": 50.0},
            simulation_backend="analytic",
            noise_model="unknown_noise",
        )


def test_sensitivity_analysis_basic() -> None:
    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import sensitivity_analysis

    model = MonoT2(te_ms=np.array([10.0, 20.0, 40.0, 80.0], dtype=np.float64))

    result = sensitivity_analysis(
        model,
        nominal_params={"m0": 1000.0, "t2_ms": 60.0},
        vary_param="t2_ms",
        lb=40.0,
        ub=100.0,
        n_steps=3,
        n_runs=2,
        simulation_backend="analytic",
        noise_model="gaussian",
        noise_snr=100.0,
        rng=np.random.default_rng(0),
    )

    assert result["vary_param"] == "t2_ms"
    assert len(result["x"]) == 3
    assert result["fit"]["t2_ms"].shape == (3, 2)
    assert "mean" in result
    assert "std" in result


def test_sensitivity_analysis_invalid_n_steps() -> None:
    import pytest

    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import sensitivity_analysis

    model = MonoT2(te_ms=np.array([10.0, 20.0], dtype=np.float64))

    with pytest.raises(ValueError, match="n_steps must be >= 2"):
        sensitivity_analysis(
            model,
            nominal_params={"m0": 100.0, "t2_ms": 50.0},
            vary_param="t2_ms",
            lb=30.0,
            ub=70.0,
            n_steps=1,
            n_runs=5,
            simulation_backend="analytic",
        )


def test_simulate_parameter_distribution_basic() -> None:
    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim import simulate_parameter_distribution

    model = MonoT2(te_ms=np.array([10.0, 20.0, 40.0], dtype=np.float64))

    result = simulate_parameter_distribution(
        model,
        true_params={"m0": np.array([900.0, 1000.0, 1100.0]), "t2_ms": 60.0},
        simulation_backend="analytic",
        noise_model="gaussian",
        noise_snr=50.0,
        rng=np.random.default_rng(123),
    )

    assert "true" in result
    assert "hat" in result
    assert "err" in result
    assert "metrics" in result
    assert result["true"]["m0"].shape == (3,)
    assert result["hat"]["t2_ms"].shape == (3,)
