#!/usr/bin/env python3
"""
Verify numerical parity between qmrpy (Python) and qMRLab (Octave).

Usage:
    uv run scripts/verify_parity.py --model mono_t2
"""
import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
# Path to qMRLab (gitignored; user must place/clone it locally if parity check is needed)
QMRLAB_PATH = ROOT_DIR / "qMRLab"
PARITY_DATA_DIR = ROOT_DIR / "data/parity"
OCTAVE_SCRIPT = ROOT_DIR / "scripts/octave/verify_models.m"


def _ensure_dir():
    PARITY_DATA_DIR.mkdir(parents=True, exist_ok=True)


def run_octave(model_name: str, input_csv: Path, output_csv: Path) -> None:
    """Run the Octave verification script."""
    cmd = [
        "octave",
        "--no-gui",
        "--eval",
        f"qMRLab_path='{QMRLAB_PATH}'; model_name='{model_name}'; input_csv='{input_csv.absolute()}'; output_csv='{output_csv.absolute()}'; source('{OCTAVE_SCRIPT.absolute()}');",
    ]
    print(f"Running Octave: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def verify_mono_t2() -> None:
    from qmrpy.models.t2 import MonoT2

    model_name = "mono_t2"
    input_csv = PARITY_DATA_DIR / f"{model_name}_input.csv"
    octave_out_csv = PARITY_DATA_DIR / f"{model_name}_octave.csv"
    
    # 1. Generate Data
    n_samples = 50
    te_ms = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    rng = np.random.default_rng(42)
    
    t2_true = rng.uniform(20, 150, n_samples)
    m0_true = rng.uniform(800, 1200, n_samples)
    
    # Create input CSV for Octave
    # Format: T2_true, M0_true, Signal(TE1)...Signal(TEn)
    # Actually qMRLab needs Signal and Protocol.
    # We will save strict columns: "id", "T2_true", "M0_true", "S_0", "S_1", ... 
    
    model_py = MonoT2(te=te_ms)
    signals = []
    for i in range(n_samples):
        s = model_py.forward(m0=m0_true[i], t2=t2_true[i])
        signals.append(s)
    signals = np.stack(signals)
    
    # Save to CSV
    df_in = pd.DataFrame({
        "id": range(n_samples),
        "t2_true": t2_true,
        "m0_true": m0_true,
    })
    for j, te in enumerate(te_ms):
        df_in[f"S_{j}"] = signals[:, j]
    
    # Save protocol as well? Octave script will need to know protocol.
    # We can pass TE in a separate file or hardcode in verify_models.m for this test.
    # For flexibility, let's write a sidecar json or just hardcode for parity check simplicity.
    # Let's write TE to a json file.
    with open(PARITY_DATA_DIR / f"{model_name}_protocol.json", "w") as f:
        json.dump({"te_ms": te_ms.tolist()}, f)
    
    df_in.to_csv(input_csv, index=False)
    
    # 2. Run Octave
    run_octave(model_name, input_csv, octave_out_csv)
    
    # 3. Load Octave Results
    # Expected columns: id, T2, M0 (fitted by Octave)
    df_oct = pd.read_csv(octave_out_csv)
    
    # 4. Run Python Model
    fitted_t2 = []
    fitted_m0 = []
    
    for i in range(n_samples):
        res = model_py.fit(signals[i])
        fitted_t2.append(res["t2"])
        fitted_m0.append(res["m0"])
        
    df_py = pd.DataFrame({
        "id": df_in["id"],
        "t2_py": fitted_t2,
        "m0_py": fitted_m0,
    })
    
    # 5. Compare
    df_py_full = pd.merge(df_py, df_in[["id", "t2_true", "m0_true"]], on="id")
    df_all = pd.merge(df_py_full, df_oct, on="id", suffixes=("", "_oct"))
    
    diff_t2 = np.abs(df_all["t2_py"] - df_all["t2_oct"])
    diff_m0 = np.abs(df_all["m0_py"] - df_all["m0_oct"])
    
    print(f"--- Verification: {model_name} ---")
    print(f"Max Diff T2: {diff_t2.max():.6e} ms")
    print(f"Max Diff M0: {diff_m0.max():.6e}")
    
    if diff_t2.max() > 1e-4:
        print("Mismatch details (Top 5 worst):")
        df_all["diff_t2"] = diff_t2
        print(df_all.sort_values("diff_t2", ascending=False)[["id", "t2_true", "t2_py", "t2_oct", "diff_t2"]].head(5))
    
    if diff_t2.max() < 1e-4:
        print("✅ T2 Parity OK")
    else:
        print("❌ T2 Parity FAIL")

    if diff_m0.max() < 1e-4:
        print("✅ M0 Parity OK")
    else:
        print("❌ M0 Parity FAIL")


def verify_vfa_t1() -> None:
    from qmrpy.models.t1 import VfaT1

    model_name = "vfa_t1"
    input_csv = PARITY_DATA_DIR / f"{model_name}_input.csv"
    octave_out_csv = PARITY_DATA_DIR / f"{model_name}_octave.csv"
    
    # 1. Generate Data
    n_samples = 50
    flip_angles = np.array([3, 10, 20, 30], dtype=float)
    tr_s = 0.015
    rng = np.random.default_rng(43)
    
    t1_true = rng.uniform(0.5, 2.0, n_samples)
    m0_true = rng.uniform(1000, 3000, n_samples)
    b1_true = rng.uniform(0.8, 1.2, n_samples)
    
    # Python Forward
    model_nominal = VfaT1(flip_angle_deg=flip_angles, tr_s=tr_s, b1=1.0)
    signals = []
    for i in range(n_samples):
        # Forward with actual B1
        model_act = VfaT1(flip_angle_deg=flip_angles, tr_s=tr_s, b1=b1_true[i])
        s = model_act.forward(m0=m0_true[i], t1_s=t1_true[i])
        signals.append(s)
    signals = np.stack(signals)
    
    # Save to CSV
    # Need to save B1 as well for Octave
    df_in = pd.DataFrame({
        "id": range(n_samples),
        "t1_true": t1_true,
        "m0_true": m0_true,
        "b1": b1_true,
    })
    for j, fa in enumerate(flip_angles):
        df_in[f"S_{j}"] = signals[:, j]
    
    with open(PARITY_DATA_DIR / f"{model_name}_protocol.json", "w") as f:
        json.dump({"flip_angle_deg": flip_angles.tolist(), "tr_s": tr_s}, f)
        
    df_in.to_csv(input_csv, index=False)
    
    # 2. Run Octave
    run_octave(model_name, input_csv, octave_out_csv)
    
    # 3. Load Octave Results
    # Expected: id, T1, M0
    # Note: qMRLab outputs T1 in seconds usually for vfa_t1.
    df_oct = pd.read_csv(octave_out_csv)
    
    # 4. Run Python Model
    fitted_t1 = []
    fitted_m0 = []
    
    for i in range(n_samples):
        # We must provide B1 to fit
        model_fit = VfaT1(flip_angle_deg=flip_angles, tr_s=tr_s, b1=b1_true[i])
        res = model_fit.fit_linear(signals[i]) # Linear fit default
        fitted_t1.append(res["t1_s"])
        fitted_m0.append(res["m0"])
        
    df_py = pd.DataFrame({
        "id": df_in["id"],
        "t1_py": fitted_t1,
        "m0_py": fitted_m0,
    })
    
    # 5. Compare
    df_all = pd.merge(df_py, df_oct, on="id", suffixes=("", "_oct"))
    
    diff_t1 = np.abs(df_all["t1_py"] - df_all["t1_oct"])
    diff_m0 = np.abs(df_all["m0_py"] - df_all["m0_oct"])
    
    print(f"--- Verification: {model_name} ---")
    print(f"Max Diff T1: {diff_t1.max():.6e} s")
    print(f"Max Diff M0: {diff_m0.max():.6e}")
    
    if diff_t1.max() < 1e-4:
        print("✅ T1 Parity OK")
    else:
        print("❌ T1 Parity FAIL")


    if diff_t1.max() < 1e-4:
        print("✅ T1 Parity OK")
    else:
        print("❌ T1 Parity FAIL")


def verify_b1_dam() -> None:
    from qmrpy.models.b1 import B1Dam

    model_name = "b1_dam"
    input_csv = PARITY_DATA_DIR / f"{model_name}_input.csv"
    octave_out_csv = PARITY_DATA_DIR / f"{model_name}_octave.csv"
    
    # 1. Generate Data
    n_samples = 50
    alpha_deg = 60.0
    rng = np.random.default_rng(44)
    
    b1_true = rng.uniform(0.6, 1.4, n_samples)
    m0_true = rng.uniform(1000, 3000, n_samples)
    
    # Python Forward
    model = B1Dam(alpha_deg=alpha_deg)
    signals = []
    for i in range(n_samples):
        s = model.forward(m0=m0_true[i], b1=b1_true[i])
        signals.append(s)
    signals = np.stack(signals)
    
    # Save to CSV
    df_in = pd.DataFrame({
        "id": range(n_samples),
        "b1_true": b1_true,
        "m0_true": m0_true,
    })
    # DAM signals: S(alpha), S(2*alpha)
    df_in["S_1"] = signals[:, 0]
    df_in["S_2"] = signals[:, 1]
    
    # qMRLab b1_dam: Prot.SEdata.Mat = [alpha_deg 2*alpha_deg] (vector of angles?)
    # or just alpha?
    # qMRLab b1_dam Prot: Alpha = deg.
    # Actually b1_dam uses DoubleAngle info. 
    # Usually: Prot.TMdata.Mat = [alpha, 2*alpha]? No, DAM is specific.
    # Looking at qMRLab b1_dam: Prot.CSdata.Mat? 
    # Let's assume we pass alpha in protocol.
    
    with open(PARITY_DATA_DIR / f"{model_name}_protocol.json", "w") as f:
        json.dump({"alpha_deg": alpha_deg}, f)
        
    df_in.to_csv(input_csv, index=False)
    
    # 2. Run Octave
    run_octave(model_name, input_csv, octave_out_csv)
    
    # 3. Load Octave Results
    # Expected: id, B1_map
    df_oct = pd.read_csv(octave_out_csv)
    
    # 4. Run Python Model
    fitted_b1 = []
    for i in range(n_samples):
        res = model.fit_raw(signals[i])
        fitted_b1.append(res["b1_raw"])
        
    df_py = pd.DataFrame({
        "id": df_in["id"],
        "b1_py": fitted_b1,
    })
    
    # 5. Compare
    df_all = pd.merge(df_py, df_oct, on="id", suffixes=("", "_oct"))
    
    diff_b1 = np.abs(df_all["b1_py"] - df_all["b1_oct"])
    
    print(f"--- Verification: {model_name} ---")
    print(f"Max Diff B1: {diff_b1.max():.6e}")
    
    if diff_b1.max() < 1e-4:
        print("✅ B1 Parity OK")
    else:
        print("❌ B1 Parity FAIL")


    if diff_b1.max() < 1e-4:
        print("✅ B1 Parity OK")
    else:
        print("❌ B1 Parity FAIL")


def verify_inversion_recovery() -> None:
    from qmrpy.models.t1 import InversionRecovery

    model_name = "inversion_recovery"
    input_csv = PARITY_DATA_DIR / f"{model_name}_input.csv"
    octave_out_csv = PARITY_DATA_DIR / f"{model_name}_octave.csv"
    
    # 1. Generate Data
    n_samples = 50
    ti_ms = np.array([50, 100, 200, 400, 800, 1600, 3000], dtype=float)
    rng = np.random.default_rng(45)
    
    t1_true = rng.uniform(500, 2000, n_samples)
    m0_true = rng.uniform(1000, 3000, n_samples)
    # in qMRLab IR: S = M0 * (1 - 2*exp(-TI/T1)) ideally, or general Barral
    # My InversionRecovery supports ra, rb.
    # Barral: S = ra + rb * exp(-TI/T1)
    # Ideally ra ~ M0, rb ~ -2M0
    ra = m0_true
    rb = -2.0 * m0_true
    
    model = InversionRecovery(ti_ms=ti_ms)
    signals = []
    for i in range(n_samples):
        s = model.forward(t1_ms=t1_true[i], ra=ra[i], rb=rb[i], magnitude=False)
        signals.append(s)
    signals = np.stack(signals)
    
    # Save to CSV
    df_in = pd.DataFrame({
        "id": range(n_samples),
        "t1_true": t1_true,
        "m0_true": m0_true,
    })
    for j, ti in enumerate(ti_ms):
        df_in[f"S_{j}"] = signals[:, j]
    
    with open(PARITY_DATA_DIR / f"{model_name}_protocol.json", "w") as f:
        json.dump({"ti_ms": ti_ms.tolist()}, f)
    
    df_in.to_csv(input_csv, index=False)
    
    # 2. Run Octave
    run_octave(model_name, input_csv, octave_out_csv)
    
    # 3. Load Octave Results
    # Expected: id, T1, M0
    # Note: qMRLab inversion recovery typically returns T1 (ms), ra, rb.
    df_oct = pd.read_csv(octave_out_csv)
    
    # 4. Run Python Model
    fitted_t1 = []
    for i in range(n_samples):
        # We assume magnitude fit often used, but here we used complex signal?
        # Let's simple fit complex for parity if possible, or magnitude
        # qMRLab inversion_recovery uses magnitude usually with Barral?
        # Let's try complex fit if signal is signed.
        # My InversionRecovery supports 'complex' method.
        res = model.fit(signals[i], method="complex")
        fitted_t1.append(res["t1_ms"])
        
    df_py = pd.DataFrame({
        "id": df_in["id"],
        "t1_py": fitted_t1,
    })
    
    # 5. Compare
    df_all = pd.merge(df_py, df_oct, on="id", suffixes=("", "_oct"))
    
    diff_t1 = np.abs(df_all["t1_py"] - df_all["t1_oct"])
    
    print(f"--- Verification: {model_name} ---")
    print(f"Max Diff T1: {diff_t1.max():.6e} ms")
    
    if diff_t1.max() < 1e-3: # slightly looser for non-linear fit differences
        print("✅ T1 Parity OK")
    else:
        print("❌ T1 Parity FAIL")


    if diff_t1.max() < 1e-3: # slightly looser for non-linear fit differences
        print("✅ T1 Parity OK")
    else:
        print("❌ T1 Parity FAIL")


def verify_mwf() -> None:
    from qmrpy.models.t2.mwf import MultiComponentT2

    model_name = "mwf"
    input_csv = PARITY_DATA_DIR / f"{model_name}_input.csv"
    octave_out_csv = PARITY_DATA_DIR / f"{model_name}_octave.csv"
    
    # 1. Generate Data
    n_samples = 50
    # Standard 32 echoes CP-MG
    te_ms = np.linspace(10, 320, 32)
    rng = np.random.default_rng(46)
    
    mwf_true = rng.uniform(0.05, 0.25, n_samples)
    m0 = 1000.0
    t2_myelin = 20.0
    t2_ie = 80.0
    
    signals = []
    # Simple 2-pool simulation
    for i in range(n_samples):
        # S = M0 * (MWF * exp(-TE/T2s) + (1-MWF) * exp(-TE/T2l))
        # Note: qMRLab might use different definition or basis.
        # But NNLS should capture it if basis covers 20ms and 80ms.
        s = m0 * (mwf_true[i] * np.exp(-te_ms / t2_myelin) + (1.0 - mwf_true[i]) * np.exp(-te_ms / t2_ie))
        signals.append(s)
    signals = np.stack(signals)
    
    # Save to CSV
    df_in = pd.DataFrame({
        "id": range(n_samples),
        "mwf_true": mwf_true,
    })
    for j, val in enumerate(te_ms):
        df_in[f"S_{j}"] = signals[:, j]
    
    with open(PARITY_DATA_DIR / f"{model_name}_protocol.json", "w") as f:
        json.dump({"te_ms": te_ms.tolist()}, f)
    
    df_in.to_csv(input_csv, index=False)
    
    # 2. Run Octave
    run_octave(model_name, input_csv, octave_out_csv)
    
    # 3. Load Octave Results
    # Expected: id, MWF
    df_oct = pd.read_csv(octave_out_csv)
    
    # 4. Run Python Model
    fitted_mwf = []
    
    # Initialize model with default basis (10ms-2000ms), 
    # check if 20ms and 80ms are covered roughly.
    model = MultiComponentT2(te_ms=te_ms)
    
    for i in range(n_samples):
        res = model.fit(signals[i], regularization_alpha=0.001) # small reg
        fitted_mwf.append(res["mwf"])
        
    df_py = pd.DataFrame({
        "id": df_in["id"],
        "mwf_py": fitted_mwf,
    })
    
    # 5. Compare
    df_all = pd.merge(df_py, df_oct, on="id", suffixes=("", "_oct"))
    
    # Fix scaling: qMRLab uses %, we use fraction
    df_all["mwf_oct"] = df_all["mwf_oct"] / 100.0
    
    diff_mwf = np.abs(df_all["mwf_py"] - df_all["mwf_oct"])
    
    print(f"--- Verification: {model_name} ---")
    print(f"Max Diff MWF: {diff_mwf.max():.6e}")
    print("Note: MWF parity depends on basis set and regularization match.")
    
    if diff_mwf.max() < 0.05: # Allow some diff due to implementation details (reg, basis)
        print("✅ MWF Parity OK (within 5%)")
    else:
        print("❌ MWF Parity FAIL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mono_t2", "vfa_t1", "b1_dam", "inversion_recovery", "mwf"])
    args = parser.parse_args()
    
    _ensure_dir()
    
    if args.model == "mono_t2":
        verify_mono_t2()
    elif args.model == "vfa_t1":
        verify_vfa_t1()
    elif args.model == "b1_dam":
        verify_b1_dam()
    elif args.model == "inversion_recovery":
        verify_inversion_recovery()
    elif args.model == "mwf":
        verify_mwf()
    else:
        print("Not implemented yet")

if __name__ == "__main__":
    main()
