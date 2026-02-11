def test_emc_t2_fit_recovers_t2():
    import pytest

    pytest.importorskip("numpy")

    from qmrpy.models import T2EMC

    model = T2EMC(n_te=16, te_ms=10.0, t1_ms=1000.0)
    signal = model.forward(m0=1000.0, t2_ms=80.0, b1=1.0)

    out = model.fit(signal)
    assert abs(out["params"]["t2_ms"] - 80.0) / 80.0 < 0.1


def test_epg_t2_estimate_b1_mode():
    import pytest

    pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2EPG

    model = T2EPG(n_te=16, te_ms=10.0, t1_ms=1000.0, alpha_deg=120.0)
    signal = model.forward(m0=1000.0, t2_ms=70.0, b1=0.85)

    out = model.fit(signal, estimate_b1=True)
    assert abs(out["params"]["t2_ms"] - 70.0) / 70.0 < 0.1
    assert abs(out["params"]["b1"] - 0.85) < 0.1


def test_epg_t2_fit_image_estimate_b1_outputs_map():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from qmrpy.models import T2EPG

    model = T2EPG(n_te=12, te_ms=10.0, t1_ms=1000.0)
    s1 = model.forward(m0=1000.0, t2_ms=70.0, b1=0.9)
    s2 = model.forward(m0=1000.0, t2_ms=70.0, b1=1.0)
    img = np.stack([s1, s2], axis=0).reshape(2, 1, -1)

    out = model.fit_image(img, estimate_b1=True)
    assert "b1" in out["params"]
    assert out["params"]["b1"].shape == (2, 1)
