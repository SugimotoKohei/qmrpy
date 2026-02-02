from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

def generate_4d_phantom(
    sx: int = 20, 
    sy: int = 20, 
    sz: int = 10, 
    n_vol: int = 30, 
    snr: float = 20.0, 
    seed: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Generate a simple 4D diffusion-like phantom with Rician noise.

    Parameters
    ----------
    sx, sy, sz : int, optional
        Spatial dimensions.
    n_vol : int, optional
        Number of volumes.
    snr : float, optional
        Target SNR.
    seed : int or None, optional
        Random seed.

    Returns
    -------
    noisy_data : ndarray
        Noisy 4D data.
    ground_truth : ndarray
        Noise-free 4D data.
    sigma : float
        Noise standard deviation.
    """
    rng = np.random.default_rng(seed)
    
    # Structure: A box in the middle
    ground_truth = np.zeros((sx, sy, sz, n_vol), dtype=np.float64)
    
    # Signal = S0 * exp(-b * D)
    b_values = np.linspace(0, 1000, n_vol)
    
    # Tissue A (Center box)
    cx, cy = sx // 2, sy // 2
    r = sx // 4
    
    # Create mask for tissue
    # Ensure indices are valid
    x0, x1 = max(0, cx-r), min(sx, cx+r)
    y0, y1 = max(0, cy-r), min(sy, cy+r)
    z0, z1 = 1, sz-1 # Skip top/bottom slices to emulate background
    
    if z1 > z0:
        mask_a = np.zeros((sx, sy, sz), dtype=bool)
        mask_a[x0:x1, y0:y1, z0:z1] = True
    else:
        # Fallback for very small Z
        mask_a = np.ones((sx, sy, sz), dtype=bool)

    adc_a = 0.001 # mm2/s
    s0_a = 1000.0
    sig_a = s0_a * np.exp(-b_values * adc_a)
    
    # Assign signal to masked region
    # Broadcast sig_a (n_vol,) to (N_mask, n_vol)
    ground_truth[mask_a] = sig_a
    
    # Add Rician noise
    # Signal = sqrt( (real + n1)^2 + (imag + n2)^2 )
    # If starting from magnitude S:
    # Real = S + n1, Imag = n2  (Approximation for S > 0, assuming phase=0)
    sigma = s0_a / snr
    
    n1 = rng.normal(0, sigma, size=ground_truth.shape)
    n2 = rng.normal(0, sigma, size=ground_truth.shape)
    
    noisy_data = np.sqrt((ground_truth + n1)**2 + n2**2)
    
    return noisy_data, ground_truth, sigma


def _shepp_logan_ellipses(*, modified: bool = True) -> list[tuple[float, float, float, float, float, float]]:
    if modified:
        # (A, a, b, x0, y0, phi_deg)
        return [
            (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
            (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
            (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18.0),
            (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18.0),
            (0.1, 0.2100, 0.2500, 0.0, 0.35, 0.0),
            (0.1, 0.0460, 0.0460, 0.0, 0.10, 0.0),
            (0.1, 0.0460, 0.0460, 0.0, -0.10, 0.0),
            (0.1, 0.0460, 0.0230, -0.08, -0.605, 0.0),
            (0.1, 0.0230, 0.0230, 0.0, -0.606, 0.0),
            (0.1, 0.0230, 0.0460, 0.06, -0.605, 0.0),
        ]
    return [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.98, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
        (-0.02, 0.1100, 0.3100, 0.22, 0.0, -18.0),
        (-0.02, 0.1600, 0.4100, -0.22, 0.0, 18.0),
        (0.01, 0.2100, 0.2500, 0.0, 0.35, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, 0.10, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, -0.10, 0.0),
        (0.01, 0.0460, 0.0230, -0.08, -0.605, 0.0),
        (0.01, 0.0230, 0.0230, 0.0, -0.606, 0.0),
        (0.01, 0.0230, 0.0460, 0.06, -0.605, 0.0),
    ]


def _shepp_logan_ellipsoids_3d(
    *,
    modified: bool = True,
    z_scale: float = 1.0,
) -> list[tuple[float, float, float, float, float, float, float, float]]:
    """Simple 3D extension of the 2D Shepp-Logan ellipses.

    Each 2D ellipse is extruded into a 3D ellipsoid by assigning a z-radius
    proportional to its y-radius (b * z_scale) and centering at z=0.
    """
    ellipses_2d = _shepp_logan_ellipses(modified=modified)
    ellipsoids: list[tuple[float, float, float, float, float, float, float, float]] = []
    z_scale = float(z_scale)
    for amp, a, b, x0, y0, phi_deg in ellipses_2d:
        c = float(b) * z_scale
        z0 = 0.0
        ellipsoids.append((float(amp), float(a), float(b), float(c), float(x0), float(y0), z0, float(phi_deg)))
    return ellipsoids


def shepp_logan_2d(
    nx: int = 128,
    ny: int = 128,
    *,
    modified: bool = True,
    ellipses: Sequence[tuple[float, float, float, float, float, float]] | None = None,
) -> NDArray[np.float64]:
    """Generate a 2D Shepp-Logan phantom (modified by default)."""
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    if ellipses is None:
        ellipses = _shepp_logan_ellipses(modified=modified)

    x = np.linspace(-1.0, 1.0, int(nx), dtype=np.float64)
    # Match array row direction to image coordinates (top -> bottom).
    y = np.linspace(1.0, -1.0, int(ny), dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    phantom = np.zeros((ny, nx), dtype=np.float64)
    for amp, a, b, x0, y0, phi_deg in ellipses:
        phi = np.deg2rad(phi_deg)
        x_rot = (xx - x0) * np.cos(phi) + (yy - y0) * np.sin(phi)
        y_rot = -(xx - x0) * np.sin(phi) + (yy - y0) * np.cos(phi)
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
        phantom[mask] += float(amp)

    return phantom


def shepp_logan_3d(
    nx: int = 128,
    ny: int = 128,
    nz: int = 64,
    *,
    modified: bool = True,
    ellipsoids: Sequence[tuple[float, float, float, float, float, float, float, float]] | None = None,
    z_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a 3D Shepp-Logan phantom (simple 3D extension by default)."""
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive")
    if ellipsoids is None:
        ellipsoids = _shepp_logan_ellipsoids_3d(modified=modified, z_scale=z_scale)

    x = np.linspace(-1.0, 1.0, int(nx), dtype=np.float64)
    # Match array row direction to image coordinates (top -> bottom).
    y = np.linspace(1.0, -1.0, int(ny), dtype=np.float64)
    z = np.linspace(1.0, -1.0, int(nz), dtype=np.float64)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    phantom = np.zeros((nz, ny, nx), dtype=np.float64)
    for amp, a, b, c, x0, y0, z0, phi_deg in ellipsoids:
        phi = np.deg2rad(phi_deg)
        x_rot = (xx - x0) * np.cos(phi) + (yy - y0) * np.sin(phi)
        y_rot = -(xx - x0) * np.sin(phi) + (yy - y0) * np.cos(phi)
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 + ((zz - z0) / c) ** 2 <= 1.0
        phantom[mask] += float(amp)

    return phantom


def shepp_logan_2d_maps(
    nx: int = 128,
    ny: int = 128,
    *,
    t1_range_ms: tuple[float, float] = (600.0, 1400.0),
    t2_range_ms: tuple[float, float] = (40.0, 120.0),
    pd_max: float = 1.0,
    modified: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate PD/T1/T2 maps from a 2D Shepp-Logan phantom."""
    phantom = shepp_logan_2d(nx=nx, ny=ny, modified=modified)
    pd = np.clip(phantom, 0.0, None)
    if np.max(pd) > 0:
        pd = pd / float(np.max(pd))
    pd = pd * float(pd_max)

    t1_min, t1_max = float(t1_range_ms[0]), float(t1_range_ms[1])
    t2_min, t2_max = float(t2_range_ms[0]), float(t2_range_ms[1])
    t1_ms = t1_min + (t1_max - t1_min) * pd
    t2_ms = t2_min + (t2_max - t2_min) * pd

    t1_ms = np.where(pd > 0, t1_ms, 0.0)
    t2_ms = np.where(pd > 0, t2_ms, 0.0)
    return pd, t1_ms, t2_ms


def shepp_logan_3d_maps(
    nx: int = 128,
    ny: int = 128,
    nz: int = 64,
    *,
    t1_range_ms: tuple[float, float] = (600.0, 1400.0),
    t2_range_ms: tuple[float, float] = (40.0, 120.0),
    pd_max: float = 1.0,
    modified: bool = True,
    z_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate PD/T1/T2 maps from a 3D Shepp-Logan phantom."""
    phantom = shepp_logan_3d(nx=nx, ny=ny, nz=nz, modified=modified, z_scale=z_scale)
    pd = np.clip(phantom, 0.0, None)
    if np.max(pd) > 0:
        pd = pd / float(np.max(pd))
    pd = pd * float(pd_max)

    t1_min, t1_max = float(t1_range_ms[0]), float(t1_range_ms[1])
    t2_min, t2_max = float(t2_range_ms[0]), float(t2_range_ms[1])
    t1_ms = t1_min + (t1_max - t1_min) * pd
    t2_ms = t2_min + (t2_max - t2_min) * pd

    t1_ms = np.where(pd > 0, t1_ms, 0.0)
    t2_ms = np.where(pd > 0, t2_ms, 0.0)
    return pd, t1_ms, t2_ms
