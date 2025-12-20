def test_qsm_split_bregman_shapes():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.qsm import qsm_split_bregman, calc_chi_l2

    shape = (6, 6, 6)
    nfm = np.random.default_rng(0).normal(0, 1, size=shape)
    mask = np.ones(shape, dtype=float)

    chi = qsm_split_bregman(
        nfm,
        mask,
        lambda_l1=1e-3,
        lambda_l2=1e-2,
        direction="forward",
        image_resolution_mm=np.array([1.0, 1.0, 1.0]),
        pad_size=(1, 1, 1),
        precon_mag_weight=False,
    )
    assert chi.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)

    chi_l2, chi_l2_pcg = calc_chi_l2(
        nfm,
        lambda_l2=1e-2,
        direction="forward",
        image_resolution_mm=np.array([1.0, 1.0, 1.0]),
        mask=mask,
        padding_size=(1, 1, 1),
    )
    assert chi_l2.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)
    assert chi_l2_pcg is None
