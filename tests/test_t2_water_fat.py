from __future__ import annotations


def test_t2_water_fat_forward_and_fit_recovers_grid_point():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2WaterFat

    te_ms = np.array([10.0, 20.0, 40.0, 80.0, 120.0, 160.0], dtype=float)
    model = T2WaterFat(te_ms=te_ms)
    signal = model.forward(
        water_amplitude=800.0,
        fat_amplitude=200.0,
        water_t2_ms=80.0,
        fat_t2_ms=35.0,
    )

    out = model.fit(signal, water_t2_grid_ms=[60.0, 80.0, 100.0], fat_t2_grid_ms=[25.0, 35.0, 45.0])

    assert out["water_t2_ms"] == pytest.approx(80.0)
    assert out["fat_t2_ms"] == pytest.approx(35.0)
    assert out["water_amplitude"] == pytest.approx(800.0, rel=1e-10)
    assert out["fat_amplitude"] == pytest.approx(200.0, rel=1e-10)
    assert out["fat_fraction"] == pytest.approx(0.2, rel=1e-10)
    assert out["quality"]["rmse"] < 1e-8


def test_t2_water_fat_fit_image_supports_mask_and_parallel():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2WaterFat

    model = T2WaterFat(te_ms=[10.0, 20.0, 40.0, 80.0, 120.0, 160.0])
    signal_a = model.forward(water_amplitude=800.0, fat_amplitude=200.0, water_t2_ms=80.0, fat_t2_ms=35.0)
    signal_b = model.forward(water_amplitude=600.0, fat_amplitude=400.0, water_t2_ms=100.0, fat_t2_ms=45.0)
    image = np.stack([signal_a, signal_b], axis=0).reshape(2, 1, -1)
    mask = np.array([[True], [False]], dtype=bool)

    out = model.fit_image(
        image,
        water_t2_grid_ms=[80.0, 100.0],
        fat_t2_grid_ms=[35.0, 45.0],
        mask=mask,
        n_jobs=-1,
    )

    assert out["fat_fraction"].shape == image.shape[:-1]
    assert out["fat_fraction"][0, 0] == pytest.approx(0.2, rel=1e-10)
    assert np.isnan(out["fat_fraction"][1, 0])


def test_t2_water_fat_functional_wrappers():
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy import fit_t2_water_fat, simulate_t2_water_fat

    te_ms = [10.0, 20.0, 40.0, 80.0, 120.0, 160.0]
    signal = simulate_t2_water_fat(
        te_ms=te_ms,
        water_amplitude=700.0,
        fat_amplitude=300.0,
        water_t2_ms=90.0,
        fat_t2_ms=40.0,
    )
    out = fit_t2_water_fat(
        signal,
        te_ms=te_ms,
        water_t2_grid_ms=[70.0, 90.0],
        fat_t2_grid_ms=[30.0, 40.0],
    )

    assert out["water_t2_ms"] == pytest.approx(90.0)
    assert out["fat_t2_ms"] == pytest.approx(40.0)
    assert out["fat_fraction"] == pytest.approx(0.3, rel=1e-10)


def test_t2_water_fat_rejects_invalid_inputs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import T2WaterFat

    with pytest.raises(ValueError, match="must not be empty"):
        T2WaterFat(te_ms=[])
    with pytest.raises(ValueError, match="non-negative"):
        T2WaterFat(te_ms=[10.0, -20.0])

    model = T2WaterFat(te_ms=[10.0, 20.0])
    with pytest.raises(ValueError, match="signal shape"):
        model.fit([1.0], water_t2_grid_ms=[80.0], fat_t2_grid_ms=[30.0])
    with pytest.raises(ValueError, match="mask must be None"):
        model.fit_image([1.0, 0.8], water_t2_grid_ms=[80.0], fat_t2_grid_ms=[30.0], mask=np.array([True]))
