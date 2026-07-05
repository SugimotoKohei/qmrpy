from __future__ import annotations


def test_mrf_dictionary_forward_and_fit_recovers_grid_point():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MRFDictionary

    flip_angle_deg = np.array([5.0, 25.0, 10.0, 35.0, 15.0, 30.0], dtype=float)
    tr_ms = np.array([12.0, 14.0, 11.0, 16.0, 13.0, 15.0], dtype=float)
    te_ms = np.array([3.0, 6.0, 4.0, 8.0, 5.0, 7.0], dtype=float)
    model = MRFDictionary(flip_angle_deg=flip_angle_deg, tr_ms=tr_ms, te_ms=te_ms)
    signal = model.forward(m0=900.0, t1_ms=1000.0, t2_ms=80.0)

    out = model.fit(
        signal,
        t1_grid_ms=[800.0, 1000.0, 1200.0],
        t2_grid_ms=[60.0, 80.0, 100.0],
    )

    assert out["t1_ms"] == pytest.approx(1000.0)
    assert out["t2_ms"] == pytest.approx(80.0)
    assert out["m0"] == pytest.approx(900.0, rel=1e-12)
    assert out["diagnostics"]["correlation"] == pytest.approx(1.0, rel=1e-12)


def test_mrf_dictionary_fit_image_supports_mask_and_parallel():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MRFDictionary

    model = MRFDictionary(
        flip_angle_deg=[5.0, 25.0, 10.0, 35.0, 15.0, 30.0],
        tr_ms=[12.0, 14.0, 11.0, 16.0, 13.0, 15.0],
        te_ms=[3.0, 6.0, 4.0, 8.0, 5.0, 7.0],
    )
    signal_a = model.forward(m0=1000.0, t1_ms=1000.0, t2_ms=80.0)
    signal_b = model.forward(m0=800.0, t1_ms=1200.0, t2_ms=100.0)
    image = np.stack([signal_a, signal_b], axis=0).reshape(2, 1, -1)
    mask = np.array([[True], [False]], dtype=bool)

    out = model.fit_image(
        image,
        t1_grid_ms=[1000.0, 1200.0],
        t2_grid_ms=[80.0, 100.0],
        mask=mask,
        n_jobs=-1,
    )

    assert out["t1_ms"].shape == image.shape[:-1]
    assert out["t1_ms"][0, 0] == pytest.approx(1000.0)
    assert out["t2_ms"][0, 0] == pytest.approx(80.0)
    assert np.isnan(out["t1_ms"][1, 0])


def test_mrf_dictionary_functional_wrappers():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy import fit_mrf_dictionary, simulate_mrf_dictionary

    flip_angle_deg = np.array([5.0, 20.0, 10.0, 30.0], dtype=float)
    tr_ms = np.array([12.0, 14.0, 11.0, 16.0], dtype=float)
    te_ms = np.array([3.0, 6.0, 4.0, 8.0], dtype=float)
    signal = simulate_mrf_dictionary(
        m0=700.0,
        t1_ms=900.0,
        t2_ms=70.0,
        flip_angle_deg=flip_angle_deg,
        tr_ms=tr_ms,
        te_ms=te_ms,
    )
    out = fit_mrf_dictionary(
        signal,
        flip_angle_deg=flip_angle_deg,
        tr_ms=tr_ms,
        te_ms=te_ms,
        t1_grid_ms=[900.0, 1100.0],
        t2_grid_ms=[70.0, 90.0],
    )

    assert out["t1_ms"] == pytest.approx(900.0)
    assert out["t2_ms"] == pytest.approx(70.0)
    assert out["m0"] == pytest.approx(700.0, rel=1e-12)


def test_mrf_dictionary_rejects_invalid_inputs():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models import MRFDictionary

    with pytest.raises(ValueError, match="must match"):
        MRFDictionary(flip_angle_deg=[5.0, 10.0], tr_ms=[12.0])
    with pytest.raises(ValueError, match="te_ms"):
        MRFDictionary(flip_angle_deg=[5.0], tr_ms=[12.0], te_ms=[13.0])

    model = MRFDictionary(flip_angle_deg=[5.0, 10.0], tr_ms=[12.0, 14.0])
    with pytest.raises(ValueError, match="signal shape"):
        model.fit([1.0], t1_grid_ms=[1000.0], t2_grid_ms=[80.0])
    with pytest.raises(ValueError, match="mask must be None"):
        model.fit_image([1.0, 0.8], t1_grid_ms=[1000.0], t2_grid_ms=[80.0], mask=np.array([True]))
