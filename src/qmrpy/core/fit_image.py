from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any
    NDArray = Any


def run_fit_image(
    *,
    signal: ArrayLike,
    fit_func: Callable[[NDArray[Any]], dict[str, float]],
    output_keys: Iterable[str],
    mask: ArrayLike | str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
    desc: str = "Fit",
) -> dict[str, NDArray[Any]]:
    import numpy as np

    from qmrpy._mask import resolve_mask
    from qmrpy._parallel import parallel_fit

    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("signal must be at least 2D with last dim as contrasts")

    spatial_shape = arr.shape[:-1]
    flat = arr.reshape((-1, arr.shape[-1]))

    resolved_mask = resolve_mask(mask, arr)
    if resolved_mask is None:
        mask_flat = np.ones((flat.shape[0],), dtype=bool)
    else:
        if resolved_mask.shape != spatial_shape:
            raise ValueError(f"mask shape {resolved_mask.shape} must match spatial shape {spatial_shape}")
        mask_flat = resolved_mask.reshape((-1,))

    return parallel_fit(
        fit_func,
        flat,
        mask_flat,
        list(output_keys),
        spatial_shape,
        n_jobs=n_jobs,
        verbose=verbose,
        desc=desc,
    )
