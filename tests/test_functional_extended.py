def test_extended_functional_wrappers():
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    import qmrpy

    te = np.array([4.0, 8.0, 12.0, 16.0], dtype=float)
    t2s_signal = 1000.0 * np.exp(-te / 25.0)

    out_r2s = qmrpy.fit_t2star_mono_r2(t2s_signal, te_ms=te)
    assert "t2star_ms" in out_r2s["params"]

    c_signal = 500.0 * np.exp(-te / 30.0) * np.exp(1j * (0.2 + 2.0 * np.pi * 10.0 * te / 1000.0))
    out_r2s_c = qmrpy.fit_t2star_complex_r2(c_signal, te_ms=te)
    assert "delta_f_hz" in out_r2s_c["params"]

    dual = np.exp(1j * np.array([0.3, 0.5], dtype=float))
    out_b0 = qmrpy.fit_b0_dual_echo(dual, te1_ms=4.0, te2_ms=6.0)
    assert "b0_hz" in out_b0["params"]

    out_b0m = qmrpy.fit_b0_multi_echo(np.angle(c_signal), te_ms=te)
    assert "phase0_rad" in out_b0m["params"]

    vfa_sig = qmrpy.simulate_t1_vfa(m0=1000.0, t1_ms=1200.0, flip_angle_deg=[3.0, 10.0, 18.0], tr_ms=18.0)
    out_hifi = qmrpy.fit_t1_despot1_hifi(vfa_sig, flip_angle_deg=[3.0, 10.0, 18.0], tr_ms=18.0)
    assert "b1" in out_hifi["params"]

    mp_model_signal = np.array([80.0, 120.0], dtype=float)
    out_mp = qmrpy.fit_t1_mp2rage(
        mp_model_signal,
        ti1_ms=700.0,
        ti2_ms=2500.0,
        alpha1_deg=4.0,
        alpha2_deg=5.0,
    )
    assert "t1_ms" in out_mp["params"]

    epg = qmrpy.simulate_t2_epg(m0=1000.0, t2_ms=70.0, n_te=8, te_ms=10.0)
    out_emc = qmrpy.fit_t2_emc(epg, n_te=8, te_ms=10.0)
    assert "t2_ms" in out_emc["params"]
