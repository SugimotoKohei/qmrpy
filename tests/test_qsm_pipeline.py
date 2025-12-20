def test_qsm_pipeline_shapes():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.qsm import QsmSplitBregman

    shape = (6, 6, 6)
    phase = np.random.default_rng(0).normal(0, 1, size=shape)
    mask = np.ones(shape, dtype=float)

    model = QsmSplitBregman(
        sharp_filter=False,
        l1_regularized=False,
        l2_regularized=True,
        no_regularization=False,
        pad_size=(1, 1, 1),
    )

    out = model.fit(phase, mask, image_resolution_mm=np.array([1.0, 1.0, 1.0]))
    assert "unwrapped_phase" in out
    assert "chiL2" in out
    assert out["chiL2"].shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)
