from __future__ import annotations

from typing import TYPE_CHECKING
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
    
    Returns:
        noisy_data, ground_truth, sigma
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
