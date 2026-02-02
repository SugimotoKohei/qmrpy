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


def test_qsm_split_bregman_backward_direction():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.qsm import qsm_split_bregman

    shape = (6, 6, 6)
    nfm = np.random.default_rng(3).normal(0, 1, size=shape)
    mask = np.ones(shape, dtype=float)

    chi = qsm_split_bregman(
        nfm,
        mask,
        lambda_l1=1e-3,
        lambda_l2=1e-2,
        direction="backward",
        image_resolution_mm=np.array([1.0, 1.0, 1.0]),
        pad_size=(1, 1, 1),
    )
    assert chi.shape == (shape[0] - 2, shape[1] - 2, shape[2] - 2)


def test_qsm_unwrap_phase_laplacian():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.qsm import unwrap_phase_laplacian

    shape = (8, 8, 8)
    # Create wrapped phase
    phase = np.random.default_rng(4).uniform(-np.pi, np.pi, size=shape)

    unwrapped = unwrap_phase_laplacian(phase)
    assert unwrapped.shape == shape
    assert np.all(np.isfinite(unwrapped))


def test_qsm_background_removal_sharp():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.qsm import background_removal_sharp

    shape = (10, 10, 10)
    phase = np.random.default_rng(5).normal(0, 1, size=shape)
    mask = np.ones(shape, dtype=float)

    nfm, mask_out = background_removal_sharp(phase, mask)
    assert nfm.shape == shape
    assert mask_out.shape == shape


def test_qsm_calc_gradient_mask():
    import pytest

    np = pytest.importorskip("numpy")

    from qmrpy.models.qsm import calc_gradient_mask_from_magnitude

    shape = (8, 8, 8)
    pad_size = (1, 1, 1)
    padded_shape = (shape[0] + 2*pad_size[0], shape[1] + 2*pad_size[1], shape[2] + 2*pad_size[2])
    magnitude = np.abs(np.random.default_rng(6).normal(100, 20, size=shape))
    mask = np.ones(padded_shape, dtype=float)

    grad_mask = calc_gradient_mask_from_magnitude(
        magnitude,
        mask,
        pad_size=pad_size,
        direction="forward",
    )
    # Returns (padded_shape, 3) for 3 gradient directions
    assert grad_mask.shape == (*padded_shape, 3)
    assert np.all(np.isfinite(grad_mask))
