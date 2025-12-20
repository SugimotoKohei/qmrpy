from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg


def _calc_fdr(shape: tuple[int, int, int], direction: str) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
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


def _kspace_kernel(fov_mm: NDArray[np.float64], shape: tuple[int, int, int]) -> NDArray[np.float64]:
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


def _apply_forward(
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


def qsm_split_bregman(
    nfm_sharp_lunwrap: NDArray[np.float64],
    mask_sharp: NDArray[np.float64],
    *,
    lambda_l1: float,
    lambda_l2: float,
    direction: str,
    image_resolution_mm: NDArray[np.float64],
    pad_size: tuple[int, int, int],
    precon_mag_weight: bool = False,
    magn_weight: NDArray[np.float64] | None = None,
    max_iter: int = 20,
    tol_pct: float = 1.0,
    pcg_tol: float = 1e-2,
    pcg_maxiter: int = 10,
) -> NDArray[np.float64]:
    """Split-Bregman QSM reconstruction (qMRLab qsmSplitBregman).

    Parameters
    ----------
    nfm_sharp_lunwrap:
        SHARP-processed and Laplacian-unwrapped phase (3D).
    mask_sharp:
        SHARP mask (3D).
    lambda_l1:
        L1 penalty.
    lambda_l2:
        L2 penalty (mu in qMRLab).
    direction:
        'forward' or 'backward'.
    image_resolution_mm:
        Voxel size in mm (x,y,z). Used to compute FOV.
    pad_size:
        Padding size used in SHARP (x,y,z). Output is cropped by this amount.
    precon_mag_weight:
        Whether to use magnitude weighting with PCG.
    magn_weight:
        Magnitude weights (shape: [x,y,z,3]). Required if precon_mag_weight=True.
    """
    if precon_mag_weight and magn_weight is None:
        raise ValueError("magn_weight is required when precon_mag_weight=True")

    nfm = np.asarray(nfm_sharp_lunwrap, dtype=np.float64)
    mask = np.asarray(mask_sharp, dtype=np.float64)
    if nfm.shape != mask.shape:
        raise ValueError("nfm_sharp_lunwrap and mask_sharp must have same shape")

    shape = nfm.shape
    fov = np.asarray(shape, dtype=np.float64) * np.asarray(image_resolution_mm, dtype=np.float64)

    fdx, fdy, fdz = _calc_fdr(shape, direction)
    cfdx, cfdy, cfdz = np.conj(fdx), np.conj(fdy), np.conj(fdz)

    d = np.fft.fftshift(_kspace_kernel(fov, shape))
    dfy = np.conj(d) * np.fft.fftn(nfm)

    e2 = np.abs(fdx) ** 2 + np.abs(fdy) ** 2 + np.abs(fdz) ** 2
    d2 = np.abs(d) ** 2

    mu = float(lambda_l2)
    threshold = float(lambda_l1) / mu

    sb_reg = 1.0 / (np.finfo(float).eps + d2 + mu * e2)

    vx = np.zeros(shape, dtype=np.complex128)
    vy = np.zeros(shape, dtype=np.complex128)
    vz = np.zeros(shape, dtype=np.complex128)
    nx = np.zeros(shape, dtype=np.complex128)
    ny = np.zeros(shape, dtype=np.complex128)
    nz = np.zeros(shape, dtype=np.complex128)

    if precon_mag_weight:
        magn_weight = np.asarray(magn_weight, dtype=np.float64)
        if magn_weight.shape != (*shape, 3):
            raise ValueError("magn_weight must have shape (x,y,z,3)")

        d_reg = d / (np.finfo(float).eps + d2 + lambda_l2 * e2)
        d_regx = np.fft.ifftn(d_reg * np.fft.fftn(nfm))
        fu = np.fft.fftn(d_regx)

        def matvec(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return _apply_forward(
                x,
                d2,
                mu,
                fdx,
                fdy,
                fdz,
                cfdx,
                cfdy,
                cfdz,
                magn_weight,
            ).ravel()

        linop = LinearOperator((fu.size, fu.size), matvec=lambda x: matvec(x))
        precond = lambda x: (sb_reg.ravel() * x)
    else:
        fu = np.zeros(shape, dtype=np.complex128)
        linop = None
        precond = None

    for _ in range(int(max_iter)):
        fu_prev = fu

        if precon_mag_weight:
            b = dfy + mu * (
                cfdx * np.fft.fftn((vx - nx) * magn_weight[..., 0])
                + cfdy * np.fft.fftn((vy - ny) * magn_weight[..., 1])
                + cfdz * np.fft.fftn((vz - nz) * magn_weight[..., 2])
            )
            fu_vec, _ = cg(
                linop,
                b.ravel(),
                x0=fu_prev.ravel(),
                atol=0.0,
                tol=pcg_tol,
                maxiter=pcg_maxiter,
                M=LinearOperator((fu.size, fu.size), matvec=precond),
            )
            fu = fu_vec.reshape(shape)

            rxu = magn_weight[..., 0] * np.fft.ifftn(fdx * fu)
            ryu = magn_weight[..., 1] * np.fft.ifftn(fdy * fu)
            rzu = magn_weight[..., 2] * np.fft.ifftn(fdz * fu)
        else:
            fu = (dfy + mu * (cfdx * np.fft.fftn(vx - nx) + cfdy * np.fft.fftn(vy - ny) + cfdz * np.fft.fftn(vz - nz))) * sb_reg
            rxu = np.fft.ifftn(fdx * fu)
            ryu = np.fft.ifftn(fdy * fu)
            rzu = np.fft.ifftn(fdz * fu)

        rox = rxu + nx
        roy = ryu + ny
        roz = rzu + nz

        vx = np.maximum(np.abs(rox) - threshold, 0) * np.sign(rox)
        vy = np.maximum(np.abs(roy) - threshold, 0) * np.sign(roy)
        vz = np.maximum(np.abs(roz) - threshold, 0) * np.sign(roz)

        nx = rox - vx
        ny = roy - vy
        nz = roz - vz

        res_change = 100.0 * np.linalg.norm(fu.ravel() - fu_prev.ravel()) / max(
            np.linalg.norm(fu.ravel()), np.finfo(float).eps
        )
        if res_change < float(tol_pct):
            break

    chi_sb = np.fft.ifftn(fu) * mask
    px, py, pz = pad_size
    return np.real(chi_sb[px : shape[0] - px, py : shape[1] - py, pz : shape[2] - pz])


def calc_chi_l2(
    phase_unwrapped: NDArray[np.float64],
    *,
    lambda_l2: float,
    direction: str,
    image_resolution_mm: NDArray[np.float64],
    mask: NDArray[np.float64],
    padding_size: tuple[int, int, int],
    magn_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """L2-regularized QSM reconstruction (qMRLab calcChiL2)."""
    nfm = np.asarray(phase_unwrapped, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    shape = nfm.shape

    fdx, fdy, fdz = _calc_fdr(shape, direction)

    fov = np.asarray(shape, dtype=np.float64) * np.asarray(image_resolution_mm, dtype=np.float64)
    d = np.fft.fftshift(_kspace_kernel(fov, shape))

    e2 = np.abs(fdx) ** 2 + np.abs(fdy) ** 2 + np.abs(fdz) ** 2
    d2 = np.abs(d) ** 2

    d_reg = d / (np.finfo(float).eps + d2 + lambda_l2 * e2)
    d_regx = np.fft.ifftn(d_reg * np.fft.fftn(nfm))

    chi_l2 = np.real(d_regx) * mask
    px, py, pz = padding_size
    chi_l2 = chi_l2[px : shape[0] - px, py : shape[1] - py, pz : shape[2] - pz]

    if magn_weight is None:
        return chi_l2, None

    magn_weight = np.asarray(magn_weight, dtype=np.float64)
    if magn_weight.shape != (*shape, 3):
        raise ValueError("magn_weight must have shape (x,y,z,3)")

    a_inv = 1.0 / (np.finfo(float).eps + d2 + lambda_l2 * e2)
    b = d * np.fft.fftn(nfm)

    def matvec(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return _apply_forward(
            x,
            d2,
            lambda_l2,
            fdx,
            fdy,
            fdz,
            np.conj(fdx),
            np.conj(fdy),
            np.conj(fdz),
            magn_weight,
        ).ravel()

    linop = LinearOperator((b.size, b.size), matvec=lambda x: matvec(x))
    precond = lambda x: (a_inv.ravel() * x)

    f_chi0 = np.fft.fftn(d_regx)
    f_chi_vec, _ = cg(
        linop,
        b.ravel(),
        x0=f_chi0.ravel(),
        atol=0.0,
        tol=1e-3,
        maxiter=20,
        M=LinearOperator((b.size, b.size), matvec=precond),
    )

    chi = f_chi_vec.reshape(shape)
    chi_l2_pcg = np.real(np.fft.ifftn(chi)) * mask
    chi_l2_pcg = chi_l2_pcg[px : shape[0] - px, py : shape[1] - py, pz : shape[2] - pz]

    return chi_l2, chi_l2_pcg
