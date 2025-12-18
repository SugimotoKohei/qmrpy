import numpy as np

from qmrpy.models.t2.decaes_t2 import DecaesT2Map, epg_decay_curve


def test_epg_matches_exponential_for_ideal_refocusing() -> None:
    # For alpha=beta=180 and very long T1, stimulated-echo effects vanish and
    # the EPG decay should closely follow a simple exponential (up to normalization).
    etl = 16
    te = 0.010
    t2 = 0.080
    t1 = 1e9

    dc = epg_decay_curve(
        etl=etl,
        alpha_deg=180.0,
        te_s=te,
        t2_s=t2,
        t1_s=t1,
        beta_deg=180.0,
        backend="epgpy",
    )
    dc_decaes = epg_decay_curve(
        etl=etl,
        alpha_deg=180.0,
        te_s=te,
        t2_s=t2,
        t1_s=t1,
        beta_deg=180.0,
        backend="decaes",
    )
    exp_ref = np.exp(-np.arange(1, etl + 1) * te / t2)

    # normalize both by first echo
    dc = dc / dc[0]
    dc_decaes = dc_decaes / dc_decaes[0]
    exp_ref = exp_ref / exp_ref[0]

    assert np.allclose(dc, exp_ref, rtol=2e-2, atol=2e-2)
    assert np.allclose(dc_decaes, exp_ref, rtol=2e-2, atol=2e-2)


def test_decaes_fit_runs() -> None:
    m = DecaesT2Map(
        n_te=16,
        te_s=0.010,
        n_t2=30,
        t2_range_s=(0.010, 2.0),
        set_flip_angle_deg=180.0,
        epg_backend="epgpy",
    )
    # synthesize as single T2 component (not exact under EPG, but should fit)
    t2_true = 0.080
    signal = epg_decay_curve(
        etl=m.n_te,
        alpha_deg=180.0,
        te_s=m.te_s,
        t2_s=t2_true,
        t1_s=m.t1_s,
        beta_deg=m.beta_deg,
        backend=m.epg_backend,
    )
    out = m.fit(signal)
    assert out["distribution"].shape == (m.n_t2,)
    assert np.isfinite(out["gdn"])
