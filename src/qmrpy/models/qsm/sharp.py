from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion


def background_removal_sharp(
    phase_lunwrap: NDArray[np.float64],
    mask_pad: NDArray[np.float64],
    *,
    filter_mode: str = "once",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Background phase removal using SHARP (qMRLab backgroundRemovalSharp)."""
    mode = str(filter_mode).lower().strip()
    if mode == "once":
        return _sharp_once(phase_lunwrap, mask_pad)
    if mode == "iterative":
        return _sharp_iterative(phase_lunwrap, mask_pad)
    raise ValueError("filter_mode must be 'once' or 'iterative'")


def _sharp_once(phase_lunwrap: NDArray[np.float64], mask_pad: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    shape = mask_pad.shape
    ksize = (9, 9, 9)
    threshold = 0.05

    del_sharp = _calc_del_kernel(ksize, shape)
    delsharp_inv = np.zeros_like(del_sharp)
    mask = np.abs(del_sharp) > threshold
    delsharp_inv[mask] = 1.0 / del_sharp[mask]

    mask_sharp = _erode_mask(mask_pad, ksize)

    phase_del = np.fft.ifftn(np.fft.fftn(phase_lunwrap) * del_sharp)
    phase_del = phase_del * mask_sharp

    phase_sharp = np.real(np.fft.ifftn(np.fft.fftn(phase_del) * delsharp_inv) * mask_sharp)
    return phase_sharp.astype(np.float64), mask_sharp.astype(np.float64)


def _sharp_iterative(phase_lunwrap: NDArray[np.float64], mask_pad: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    shape = mask_pad.shape
    threshold = 0.05
    kernel_sizes = list(range(9, 2, -2))

    phase_del_acc = np.zeros(shape, dtype=np.complex128)
    mask_prev = np.zeros(shape, dtype=np.float64)

    delsharp_inv = None
    mask_sharp = None
    for idx, k in enumerate(kernel_sizes):
        ksize = (k, k, k)
        del_sharp = _calc_del_kernel(ksize, shape)
        if idx == 0:
            delsharp_inv = np.zeros_like(del_sharp)
            mask = np.abs(del_sharp) > threshold
            delsharp_inv[mask] = 1.0 / del_sharp[mask]

        mask_sharp = _erode_mask(mask_pad, ksize)
        phase_del = np.fft.ifftn(np.fft.fftn(phase_lunwrap) * del_sharp)
        phase_del_acc = phase_del_acc + phase_del * (mask_sharp - mask_prev)
        mask_prev = mask_sharp

    assert delsharp_inv is not None
    assert mask_sharp is not None

    phase_sharp = np.real(np.fft.ifftn(np.fft.fftn(phase_del_acc) * delsharp_inv) * mask_sharp)
    return phase_sharp.astype(np.float64), mask_sharp.astype(np.float64)


def _calc_del_kernel(ksize: tuple[int, int, int], shape: tuple[int, int, int]) -> NDArray[np.complex128]:
    khsize = ((np.array(ksize) - 1) // 2).astype(int)
    a, b, c = np.meshgrid(
        np.arange(-khsize[1], khsize[1] + 1),
        np.arange(-khsize[0], khsize[0] + 1),
        np.arange(-khsize[2], khsize[2] + 1),
        indexing="xy",
    )

    kernel = (a**2 / khsize[0] ** 2 + b**2 / khsize[1] ** 2 + c**2 / khsize[2] ** 2) <= 1
    kernel = -kernel.astype(np.float64) / np.sum(kernel)
    kernel[khsize[0], khsize[1], khsize[2]] = 1.0 + kernel[khsize[0], khsize[1], khsize[2]]

    big_kernel = np.zeros(shape, dtype=np.float64)
    c1, c2, c3 = shape[0] // 2, shape[1] // 2, shape[2] // 2
    big_kernel[
        c1 - khsize[0] : c1 + khsize[0] + 1,
        c2 - khsize[1] : c2 + khsize[1] + 1,
        c3 - khsize[2] : c3 + khsize[2] + 1,
    ] = -kernel

    return np.fft.fftn(np.fft.fftshift(big_kernel))


def _erode_mask(mask_pad: NDArray[np.float64], ksize: tuple[int, int, int]) -> NDArray[np.float64]:
    erode_size = tuple(np.array(ksize) + 1)

    mask = mask_pad.astype(bool)
    # Erode along each axis with line structuring elements
    sx = np.ones((erode_size[0], 1, 1), dtype=bool)
    sy = np.ones((1, erode_size[1], 1), dtype=bool)
    sz = np.ones((1, 1, erode_size[2]), dtype=bool)

    mask = binary_erosion(mask, structure=sx)
    mask = binary_erosion(mask, structure=sy)
    mask = binary_erosion(mask, structure=sz)
    return mask.astype(np.float64)
