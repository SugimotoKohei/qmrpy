"""Parallel processing utilities for qmrpy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    NDArray = Any


def parallel_fit(
    fit_func: Callable[[NDArray[Any]], dict[str, float]],
    data_flat: NDArray[Any],
    mask_flat: NDArray[np.bool_],
    output_keys: Sequence[str],
    spatial_shape: tuple[int, ...],
    *,
    n_jobs: int = 1,
) -> dict[str, NDArray[Any]]:
    """Run voxel-wise fitting in parallel.

    Parameters
    ----------
    fit_func : callable
        Function that takes a 1D signal and returns a dict of fitted parameters.
    data_flat : ndarray
        Flattened data array of shape (n_voxels, n_echoes).
    mask_flat : ndarray
        Boolean mask of shape (n_voxels,).
    output_keys : sequence of str
        Keys for output parameter maps.
    spatial_shape : tuple
        Original spatial shape for reshaping output.
    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all CPUs.

    Returns
    -------
    dict
        Parameter maps with NaN for masked-out voxels.
    """
    import numpy as np

    # Initialize output arrays
    out: dict[str, NDArray[Any]] = {
        key: np.full(spatial_shape, np.nan, dtype=np.float64)
        for key in output_keys
    }

    indices = np.flatnonzero(mask_flat)

    if len(indices) == 0:
        return out

    # Serial execution
    if n_jobs == 1:
        for idx in indices:
            res = fit_func(data_flat[idx])
            for key in output_keys:
                if key in res:
                    out[key].flat[idx] = float(res[key])
        return out

    # Parallel execution
    from joblib import Parallel, delayed

    def _fit_single(idx: int) -> tuple[int, dict[str, float]]:
        return idx, fit_func(data_flat[idx])

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_single)(idx) for idx in indices
    )

    for idx, res in results:
        for key in output_keys:
            if key in res:
                out[key].flat[idx] = float(res[key])

    return out
