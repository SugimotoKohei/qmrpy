from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def unwrap_phase_laplacian(wrapped_phase: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unwrap phase volume using Laplacian technique (qMRLab unwrapPhaseLaplacian)."""
    phase = np.asarray(wrapped_phase, dtype=np.float64)
    n1, n2, n3 = phase.shape

    ksize = np.array([3, 3, 3])
    khsize = (ksize - 1) // 2

    kernel = np.zeros((3, 3, 3), dtype=np.float64)
    kernel[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel[:, :, 1] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
    kernel[:, :, 2] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    big_kernel = np.zeros_like(phase)
    c1, c2, c3 = n1 // 2, n2 // 2, n3 // 2
    big_kernel[
        c1 - khsize[0] : c1 + khsize[0] + 1,
        c2 - khsize[1] : c2 + khsize[1] + 1,
        c3 - khsize[2] : c3 + khsize[2] + 1,
    ] = -kernel

    del_op = np.fft.fftn(np.fft.fftshift(big_kernel))
    del_inv = np.zeros_like(del_op)
    mask = del_op != 0
    del_inv[mask] = 1.0 / del_op[mask]

    del_phase = (
        np.cos(phase) * np.fft.ifftn(np.fft.fftn(np.sin(phase)) * del_op)
        - np.sin(phase) * np.fft.ifftn(np.fft.fftn(np.cos(phase)) * del_op)
    )

    unwrapped = np.fft.ifftn(np.fft.fftn(del_phase) * del_inv)
    return np.real(unwrapped).astype(np.float64)
