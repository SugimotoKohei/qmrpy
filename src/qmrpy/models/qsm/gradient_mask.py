from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .utils import calc_fdr


def calc_gradient_mask_from_magnitude(
    magn: NDArray[np.float64],
    mask_sharp: NDArray[np.float64],
    *,
    pad_size: tuple[int, int, int],
    direction: str,
) -> NDArray[np.float64]:
    """Magnitude-gradient masks for QSM weighting (qMRLab calcGradientMaskFromMagnitudeImage)."""
    n = mask_sharp.shape
    fdx, fdy, fdz = calc_fdr(n, direction)

    magn_pad = np.pad(magn, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode="constant")
    magn_pad = magn_pad * mask_sharp
    maxv = float(np.max(magn_pad)) if np.max(magn_pad) > 0 else 1.0
    magn_pad = magn_pad / maxv

    magn_k = np.fft.fftn(magn_pad)
    magn_grad = np.stack(
        [np.fft.ifftn(magn_k * fdx), np.fft.ifftn(magn_k * fdy), np.fft.ifftn(magn_k * fdz)],
        axis=-1,
    )

    magn_weight = np.zeros_like(magn_grad, dtype=np.float64)
    mask = mask_sharp.astype(bool)

    for s in range(magn_grad.shape[-1]):
        magn_use = np.abs(magn_grad[..., s])
        values = magn_use[mask]
        if values.size == 0:
            threshold = 0.0
        else:
            values_sorted = np.sort(values)[::-1]
            idx = int(round(len(values_sorted) * 0.3))
            idx = max(0, min(idx, len(values_sorted) - 1))
            threshold = float(values_sorted[idx])
        magn_weight[..., s] = (magn_use <= threshold).astype(np.float64)

    return magn_weight
