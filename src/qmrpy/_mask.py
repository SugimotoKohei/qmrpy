"""Mask utilities for qmrpy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
else:
    ArrayLike = Any
    NDArray = Any


def resolve_mask(
    mask: ArrayLike | Literal["otsu"] | None,
    data: NDArray[Any],
) -> NDArray[np.bool_] | None:
    """Resolve mask argument to a boolean array.

    Parameters
    ----------
    mask : array-like, "otsu", or None
        - None: no mask (returns None)
        - "otsu": compute Otsu threshold on mean of last axis
        - array-like: use as boolean mask
    data : ndarray
        Input data array (used for Otsu computation).

    Returns
    -------
    ndarray or None
        Boolean mask array, or None if no masking.
    """
    import numpy as np

    if mask is None:
        return None

    if isinstance(mask, str):
        mask_lower = mask.lower().strip()
        if mask_lower == "otsu":
            return _otsu_mask(data)
        raise ValueError(f"Unknown mask type: {mask!r}. Use 'otsu' or an array.")

    return np.asarray(mask, dtype=bool)


def _otsu_mask(data: NDArray[Any]) -> NDArray[np.bool_]:
    """Compute Otsu threshold mask from data.

    Parameters
    ----------
    data : ndarray
        Input data. If multi-dimensional, mean along last axis is used.

    Returns
    -------
    ndarray
        Boolean mask where True indicates foreground.
    """
    import numpy as np

    # Use mean along last axis for multi-echo data
    if data.ndim > 2:
        img = np.mean(np.abs(data), axis=-1)
    else:
        img = np.abs(data)

    # Otsu threshold
    threshold = _otsu_threshold(img)
    return img > threshold


def _otsu_threshold(image: NDArray[Any]) -> float:
    """Compute Otsu's threshold.

    Parameters
    ----------
    image : ndarray
        Input image (2D or flattened).

    Returns
    -------
    float
        Optimal threshold value.
    """
    import numpy as np

    # Flatten and remove NaN/Inf
    flat = image.ravel()
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        return 0.0

    # Histogram (256 bins)
    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram
    hist = hist.astype(np.float64)
    hist_norm = hist / hist.sum()

    # Cumulative sums
    weight1 = np.cumsum(hist_norm)
    weight2 = np.cumsum(hist_norm[::-1])[::-1]

    # Cumulative means
    mean1 = np.cumsum(hist_norm * bin_centers)
    mean2 = (np.cumsum((hist_norm * bin_centers)[::-1])[::-1])

    # Between-class variance
    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = mean1 / weight1
        mean2 = mean2 / weight2
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Find maximum
    idx = np.nanargmax(variance)
    return float(bin_centers[idx])
