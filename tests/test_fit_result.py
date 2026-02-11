import numpy as np

from qmrpy.core.result_schema import FitResult, nest_result
from qmrpy.models.t2 import T2Mono


def test_fit_result_behaves_like_params_dict() -> None:
    model = T2Mono(te_ms=np.array([10.0, 20.0, 40.0, 80.0], dtype=float))
    signal = model.forward(m0=1000.0, t2_ms=80.0)

    result = model.fit(signal)
    assert isinstance(result, FitResult)

    # New ergonomic access (params as top-level mapping)
    assert abs(result["t2_ms"] - 80.0) / 80.0 < 1e-6
    assert abs(result["m0"] - 1000.0) / 1000.0 < 1e-6

    # Backward-compatible nested access
    assert result["params"]["t2_ms"] == result["t2_ms"]
    assert isinstance(result["quality"], dict)
    assert isinstance(result["diagnostics"], dict)

    nested = result.to_dict()
    assert set(nested.keys()) == {"params", "quality", "diagnostics"}
    assert nested["params"]["t2_ms"] == result["t2_ms"]


def test_fit_image_result_keeps_quality_and_diagnostics_attributes() -> None:
    model = T2Mono(te_ms=np.array([10.0, 20.0, 40.0, 80.0], dtype=float))
    signal = model.forward(m0=500.0, t2_ms=60.0)
    image = np.stack([signal, signal], axis=0).reshape(2, 1, -1)

    maps = model.fit_image(image, offset_term=True)
    assert isinstance(maps, FitResult)
    assert maps["t2_ms"].shape == (2, 1)
    assert maps["offset"].shape == (2, 1)
    assert maps["params"]["t2_ms"].shape == (2, 1)
    assert maps.quality["status"] == "ok"


def test_nest_result_supports_flat_and_nested_inputs() -> None:
    flat = {"m0": 1200.0, "t2_ms": 70.0, "rmse": 0.5, "solver_iter": 12}
    out_flat = nest_result(flat, param_keys=("m0", "t2_ms"))
    assert isinstance(out_flat, FitResult)
    assert out_flat["m0"] == 1200.0
    assert out_flat.quality["rmse"] == 0.5
    assert out_flat.diagnostics["solver_iter"] == 12

    nested = {
        "params": {"m0": 900.0, "t2_ms": 50.0},
        "quality": {"rmse": 0.1, "n_points": 4, "status": "ok"},
        "diagnostics": {"solver_iter": 5},
    }
    out_nested = nest_result(nested, param_keys=("m0", "t2_ms"))
    assert isinstance(out_nested, FitResult)
    assert out_nested["t2_ms"] == 50.0
    assert out_nested.quality["n_points"] == 4
    assert out_nested.diagnostics["solver_iter"] == 5
