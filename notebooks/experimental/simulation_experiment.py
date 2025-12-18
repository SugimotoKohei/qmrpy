def test_simulate_single_voxel_vfa_t1_noise_free_recovers_params():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VfaT1
    from qmrpy.sim.simulation import simulate_single_voxel

    model = VfaT1(flip_angle_deg=np.array([3.0, 8.0, 15.0, 25.0]), tr_s=0.015, b1=1.0)
    out = simulate_single_voxel(
        model,
        params={"m0": 2000.0, "t1_s": 0.9},
        noise_model="none",
        fit=True,
    )

    assert abs(out["fit"]["m0"] - 2000.0) / 2000.0 < 1e-6
    assert abs(out["fit"]["t1_s"] - 0.9) / 0.9 < 1e-6


def test_sensitivity_analysis_vfa_t1_noise_free_tracks_true_parameter():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VfaT1
    from qmrpy.sim.simulation import sensitivity_analysis

    model = VfaT1(flip_angle_deg=np.array([3.0, 8.0, 15.0, 25.0]), tr_s=0.015, b1=1.0)
    res = sensitivity_analysis(
        model,
        nominal_params={"m0": 2000.0, "t1_s": 1.0},
        vary_param="t1_s",
        lb=0.6,
        ub=1.4,
        n_steps=5,
        n_runs=1,
        noise_model="none",
        noise_sigma=0.0,
    )

    x = res["x"]
    t1_hat = res["mean"]["t1_s"]
    assert np.allclose(t1_hat, x, rtol=0, atol=1e-8)


def test_simulate_parameter_distribution_mono_t2_noise_free_linear_fit():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t2 import MonoT2
    from qmrpy.sim.simulation import simulate_parameter_distribution

    model = MonoT2(te=np.array([10.0, 20.0, 40.0, 80.0], dtype=float))
    t2_true = np.array([30.0, 60.0, 90.0], dtype=float)

    res = simulate_parameter_distribution(
        model,
        true_params={"m0": 1000.0, "t2": t2_true},
        noise_model="none",
        noise_sigma=0.0,
        fit_kwargs={"fit_type": "linear"},
    )

    assert np.allclose(res["hat"]["t2"], t2_true, rtol=0, atol=1e-8)


def test_crlb_and_protocol_grid_optimization_runs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.t1 import VfaT1
    from qmrpy.sim.simulation import crlb_cov_mean, optimize_protocol_grid

    params = {"m0": 2000.0, "t1_s": 1.0}

    def factory(fa_deg):
        return VfaT1(flip_angle_deg=np.asarray(fa_deg, dtype=float), tr_s=0.015, b1=1.0)

    candidates = [np.array([3.0, 8.0, 15.0, 25.0]), np.array([2.0, 5.0, 10.0, 18.0])]

    obj0 = crlb_cov_mean(factory(candidates[0]), params=params, variables=["m0", "t1_s"], sigma=1.0)
    assert np.isfinite(obj0) and obj0 > 0

    out = optimize_protocol_grid(
        factory,
        protocol_candidates=candidates,
        params=params,
        variables=["m0", "t1_s"],
        sigma=1.0,
    )
    assert out["best_index"] in {0, 1}
    assert len(out["objectives"]) == 2


def test_qmrlab_named_wrappers_import_and_run():
    import pytest

    np = pytest.importorskip("numpy")

    from dataclasses import dataclass

    from qmrpy.models.t1 import VfaT1
    from qmrpy.sim import SimCRLB, SimFisherMatrix, SimRnd, SimVary

    @dataclass
    class OptTable:
        xnames: list[str]
        fx: list[bool]
        st: list[float]
        lb: list[float]
        ub: list[float]

    model = VfaT1(flip_angle_deg=np.array([3.0, 8.0, 15.0, 25.0]), tr_s=0.015, b1=1.0)
    table = OptTable(
        xnames=["m0", "t1_s"],
        fx=[True, False],
        st=[2000.0, 1.0],
        lb=[2000.0, 0.8],
        ub=[2000.0, 1.2],
    )

    vary = SimVary(model, runs=1, OptTable=table, Opts={"SNR": 0})
    assert "t1_s" in vary

    rnd = SimRnd(model, {"m0": np.array([2000.0, 2000.0]), "t1_s": np.array([0.9, 1.1])}, {"SNR": 0})
    assert "RMSE" in rnd

    # Fisher/CRLB wrappers require xnames/fx like qMRLab models.
    class ModelProxy:
        xnames = ["m0", "t1_s"]
        fx = [False, False]

        def forward(self, **params):
            return model.forward(**params)

        def fit_linear(self, signal, **kwargs):
            return model.fit_linear(signal, **kwargs)

    proxy = ModelProxy()

    f = SimFisherMatrix(proxy, Prot=None, x=np.array([2000.0, 1.0]), variables=[1, 2], sigma=1.0)
    assert f.shape == (2, 2)

    F, names, crlb, fall = SimCRLB(proxy, Prot=None, xvalues=np.array([[2000.0, 1.0]]), sigma=1.0)
    assert names == ["m0", "t1_s"]
    assert np.isfinite(F)
    assert crlb.shape == (2, 2)
    assert fall.size == 2
