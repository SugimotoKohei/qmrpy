import numpy as np


def test_mppca_runs_and_returns_expected_shapes() -> None:
    from qmrpy.models.noise import MPPCA

    rng = np.random.default_rng(0)

    # Small 4D volume with enough interior voxels for a 3x3x3 kernel
    data = rng.normal(loc=100.0, scale=5.0, size=(9, 9, 9, 12)).astype(np.float64)
    mask = np.ones((9, 9, 9), dtype=bool)

    out = MPPCA(kernel=(3, 3, 3)).fit(data, mask=mask)
    assert set(out.keys()) == {"denoised", "sigma", "n_pars"}

    out2 = MPPCA(kernel=(3, 3, 3)).fit_image(data, mask=mask)
    assert set(out2.keys()) == {"denoised", "sigma", "n_pars"}

    den = out["denoised"]
    sig = out["sigma"]
    npars = out["n_pars"]

    assert den.shape == data.shape
    assert sig.shape == data.shape[:3]
    assert npars.shape == data.shape[:3]

    # Check a processed interior voxel
    cx, cy, cz = 4, 4, 4
    assert np.all(np.isfinite(den[cx, cy, cz, :]))
    assert np.isfinite(sig[cx, cy, cz])
    assert sig[cx, cy, cz] >= 0.0
    assert np.isfinite(npars[cx, cy, cz])
    assert 0.0 <= npars[cx, cy, cz] <= float(min(data.shape[-1], 3 * 3 * 3))

