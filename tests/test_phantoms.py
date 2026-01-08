def test_shepp_logan_2d_maps_shapes_and_ranges():
    import numpy as np

    from qmrpy.sim.phantoms import shepp_logan_2d, shepp_logan_2d_maps

    phantom = shepp_logan_2d(nx=64, ny=48)
    assert phantom.shape == (48, 64)

    pd, t1_ms, t2_ms = shepp_logan_2d_maps(nx=64, ny=48)
    assert pd.shape == (48, 64)
    assert t1_ms.shape == pd.shape
    assert t2_ms.shape == pd.shape
    assert np.min(pd) >= 0.0
    assert np.max(pd) <= 1.0 + 1e-6
    assert np.all(t1_ms[pd == 0] == 0.0)
    assert np.all(t2_ms[pd == 0] == 0.0)
