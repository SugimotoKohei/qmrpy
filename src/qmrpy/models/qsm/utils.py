from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def calc_fdr(shape: tuple[int, int, int], direction: str) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    """Finite difference operators in k-space (qMRLab calcFdr)."""
    ny, nx, nz = shape  # MATLAB meshgrid order was (k2,k1,k3) -> (y,x,z)
    k2, k1, k3 = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nz, dtype=np.float64),
        indexing="xy",
    )

    if direction == "forward":
        fdx = 1.0 - np.exp(-2j * np.pi * k1 / float(ny))
        fdy = 1.0 - np.exp(-2j * np.pi * k2 / float(nx))
        fdz = 1.0 - np.exp(-2j * np.pi * k3 / float(nz))
    elif direction == "backward":
        fdx = -1.0 + np.exp(2j * np.pi * k1 / float(ny))
        fdy = -1.0 + np.exp(2j * np.pi * k2 / float(nx))
        fdz = -1.0 + np.exp(2j * np.pi * k3 / float(nz))
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    return fdx, fdy, fdz


def kspace_kernel(fov_mm: NDArray[np.float64], shape: tuple[int, int, int]) -> NDArray[np.float64]:
    """Dipole kernel in k-space (qMRLab kspaceKernel)."""
    ny, nx, nz = shape
    center = np.array([ny, nx, nz], dtype=np.float64) / 2.0 + 1.0

    ky = np.arange(1, ny + 1, dtype=np.float64) - center[0]
    kx = np.arange(1, nx + 1, dtype=np.float64) - center[1]
    kz = np.arange(1, nz + 1, dtype=np.float64) - center[2]

    dkx = 1.0 / float(fov_mm[0])
    dky = 1.0 / float(fov_mm[1])
    dkz = 1.0 / float(fov_mm[2])

    kx = (kx * dkx).reshape(-1, 1, 1)
    ky = (ky * dky).reshape(1, -1, 1)
    kz = (kz * dkz).reshape(1, 1, -1)

    k2 = kx**2 + ky**2 + kz**2
    k2 = np.where(k2 == 0, np.finfo(float).eps, k2)

    kernel = 1.0 / 3.0 - (kz**2) / k2
    kernel[(kx == 0) & (ky == 0) & (kz == 0)] = 0.0
    return kernel.astype(np.float64)


def apply_forward(
    x: NDArray[np.complex128],
    d2: NDArray[np.float64],
    mu: float,
    fdx: NDArray[np.complex128],
    fdy: NDArray[np.complex128],
    fdz: NDArray[np.complex128],
    cfdx: NDArray[np.complex128],
    cfdy: NDArray[np.complex128],
    cfdz: NDArray[np.complex128],
    magn_weight: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Apply forward operator used in PCG (qMRLab applyForward)."""
    x = x.reshape(d2.shape)
    term_x = np.fft.fftn(magn_weight[..., 0] * np.fft.ifftn(fdx * x)) * cfdx
    term_y = np.fft.fftn(magn_weight[..., 1] * np.fft.ifftn(fdy * x)) * cfdy
    term_z = np.fft.fftn(magn_weight[..., 2] * np.fft.ifftn(fdz * x)) * cfdz
    y = d2 * x + mu * (term_x + term_y + term_z)
    return y
