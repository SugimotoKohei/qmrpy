#!/usr/bin/env python3
"""
Verification script for MP-PCA denoising (3D/4D).
Generates a 4D phantom, adds Rician noise, runs Python MPPCA,
calls Octave MPPCA, and compares.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio

from qmrpy.models.noise.denoising_mppca import MPPCA

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data/parity"
OCTAVE_SCRIPT_PATH = ROOT_DIR / "scripts/octave/verify_mppca.m"


def generate_phantom(sx=20, sy=20, sz=5, n_vol=30, snr=20):
    """Generate a simple 4D diffusion-like phantom."""
    np.random.seed(42)
    
    # Structure: A box in the middle
    ground_truth = np.zeros((sx, sy, sz, n_vol))
    
    # 3 regions: Background (0), Tissue A (decay fast), Tissue B (decay slow)
    # Signal = S0 * exp(-b * D)
    # Let's just simulate random exponential decays
    b_values = np.linspace(0, 1000, n_vol)
    
    # Tissue A (Center box)
    cx, cy = sx // 2, sy // 2
    r = sx // 4
    mask_a = np.zeros((sx, sy, sz), dtype=bool)
    mask_a[cx-r:cx+r, cy-r:cy+r, 1:sz-1] = True
    
    adc_a = 0.001 # mm2/s
    s0_a = 1000.0
    sig_a = s0_a * np.exp(-b_values * adc_a)
    
    # Assign
    ground_truth[mask_a] = sig_a
    
    # Add Rician noise
    sigma = s0_a / snr
    
    # Rician = sqrt( (S + n1)^2 + n2^2 )
    n1 = np.random.normal(0, sigma, size=ground_truth.shape)
    n2 = np.random.normal(0, sigma, size=ground_truth.shape)
    
    noisy_data = np.sqrt((ground_truth + n1)**2 + n2**2)
    
    return noisy_data, ground_truth, sigma


def run_octave(mat_path_in, mat_path_out):
    """Call Octave script to run MP-PCA."""
    qMRLab_path = ROOT_DIR / "qMRLab"
    
    cmd = [
        "octave",
        "--no-gui",
        "--eval",
        f"qMRLab_path='{qMRLab_path}'; input_mat='{mat_path_in}'; output_mat='{mat_path_out}'; source('{OCTAVE_SCRIPT_PATH}');"
    ]
    print(f"Running Octave...")
    subprocess.check_call(cmd)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Data
    print("Generating 4D phantom...")
    noisy, true_sig, true_sigma = generate_phantom(sx=20, sy=20, sz=10, n_vol=30, snr=20)
    
    # Save for Octave
    input_mat = DATA_DIR / "mppca_input.mat"
    output_mat = DATA_DIR / "mppca_octave.mat"
    
    sio.savemat(input_mat, {"Data4D": noisy})
    
    # 2. Run Python MPPCA
    print("Running Python MPPCA...")
    model = MPPCA(kernel=(5, 5, 5))
    res_py = model.fit(noisy)
    denoised_py = res_py["denoised"]
    sigma_py = res_py["sigma"]
    
    # 3. Run Octave MPPCA
    run_octave(input_mat, output_mat)
    
    # Load Octave results
    if not output_mat.exists():
        print("Octave output not found!")
        return
        
    res_oct = sio.loadmat(output_mat)
    denoised_oct = res_oct["denoised"]
    sigma_oct = res_oct["sigma"]
    
    # 4. Compare
    print("\n--- Verification: MP-PCA ---")
    
    # Compare center values (avoid boundaries)
    mid_x, mid_y, mid_z = 10, 10, 5
    print(f"Center Pixel ({mid_x},{mid_y},{mid_z}):")
    print(f"  Py Sigma: {sigma_py[mid_x, mid_y, mid_z]:.4f}")
    print(f"  Oct Sigma: {sigma_oct[mid_x, mid_y, mid_z]:.4f}")
    print(f"  Py Signal[0]: {denoised_py[mid_x, mid_y, mid_z, 0]:.4f}")
    print(f"  Oct Signal[0]: {denoised_oct[mid_x, mid_y, mid_z, 0]:.4f}")

    # ROI Comparison (exclude boundaries and qMRLab weirdness)
    # qMRLab zeros out boundaries widely.
    # We crop 3 pixels from each side (kernel half-width is 2, plus margin)
    sl = slice(3, -3)
    # Z-dimension is small (10). 3:-3 leaves 4 slices (3,4,5,6).
    roi = (sl, sl, sl)
    
    d_py_roi = denoised_py[roi]
    d_oct_roi = denoised_oct[roi]
    s_py_roi = sigma_py[roi]
    s_oct_roi = sigma_oct[roi]
    
    diff_sig = np.abs(d_py_roi - d_oct_roi)
    max_diff_sig = np.max(diff_sig)
    print(f"\nROI Max Diff Signal: {max_diff_sig:.6e}")
    
    diff_sigma = np.abs(s_py_roi - s_oct_roi)
    max_diff_sigma = np.max(diff_sigma)
    print(f"ROI Max Diff Sigma: {max_diff_sigma:.6e}")

    # Metrics
    if max_diff_sig < 1e-4:
        print("✅ Signal Parity OK")
    else:
        print("❌ Signal Parity FAIL")
        
    if max_diff_sigma < 1e-4:
        print("✅ Sigma Parity OK")
    else:
        print("❌ Sigma Parity FAIL")
    print(f"Center Pixel ({mid_x},{mid_y},{mid_z}):")
    print(f"  Py Sigma: {sigma_py[mid_x, mid_y, mid_z]:.4f}")
    print(f"  Oct Sigma: {sigma_oct[mid_x, mid_y, mid_z]:.4f}")
    print(f"  Py n_pars: {res_py['n_pars'][mid_x, mid_y, mid_z]:.1f}")
    print(f"  Py Signal[0]: {denoised_py[mid_x, mid_y, mid_z, 0]:.4f}")
    print(f"  Oct Signal[0]: {denoised_oct[mid_x, mid_y, mid_z, 0]:.4f}")

    # Metrics
    if max_diff_sig < 1e-4:
        print("✅ Signal Parity OK")
    else:
        print("❌ Signal Parity FAIL")
        
    if max_diff_sigma < 1e-4:
        print("✅ Sigma Parity OK")
    else:
        print("❌ Sigma Parity FAIL")


if __name__ == "__main__":
    main()
