import nbformat as nbf

nb = nbf.v4.new_notebook()

# Title
text = """# MRzero Simulation Tutorial: From Bloch Basics to QSM

This tutorial provides a comprehensive guide to MRI simulations using **MRzeroCore** and **qmrpy**.
It covers fundamental physics and quantitative imaging applications.

**Table of Contents:**
1.  **Bloch Simulation Basics:** Spin Echo, T2 Decay, and Stimulated Echoes.
2.  **T1 Mapping:** VFA (SPGR), B1 Inhomogeneity, and AFI Correction.
3.  **T2 Mapping & MWF:** Multi-component T2 simulation and Myelin Water Fraction estimation.
4.  **T2* & Dephasing:** Reversible (Spin Echo) vs Irreversible (FID) dephasing under B0 inhomogeneity.
5.  **QSM:** 3D Susceptibility Mapping simulation and reconstruction.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

# Imports
code = """import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, geom_point, labs, theme_bw, scale_color_manual, geom_col, scale_x_log10, geom_hline, scale_linetype_manual, geom_vline
from pathlib import Path

import MRzeroCore as mr0
from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.calc_duration import calc_duration
from pypulseq.opts import Opts

from qmrpy.sim import simulate_bloch
from qmrpy.models.t1 import T1VFA
from qmrpy.models.b1 import B1AFI
from qmrpy.models.t2 import T2Mono
try:
    from qmrpy.models.t2 import T2MultiComponent
except ImportError:
    from qmrpy.models.t2.mwf import T2MultiComponent
from qmrpy.models.qsm.pipeline import QSMSplitBregman
from qmrpy.models.qsm.utils import kspace_kernel

# Setup Output Directory
Path("output/seq").mkdir(parents=True, exist_ok=True)
"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# --- Chapter 1 ---
text = """## 1. Bloch Simulation Basics: Spin Echo & Stimulated Echoes

We simulate a single voxel using a CPMG sequence.
We compare an ideal 180° refocusing pulse (pure Spin Echo) with a 60° refocusing pulse (generating Stimulated Echoes).
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code = """# System & Sequence Helper
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

def quantize(t, raster):
    return float(np.round(t / raster) * raster)

def build_cpmg_seq(te=10e-3, n_echo=16, adc_samples=1, refoc_angle_deg=180.0):
    seq = Sequence(system)
    phase_exc = 0.0
    phase_ref = np.pi / 2.0
    
    # 90 deg Excitation + Rewinder
    rf90, gz90, _ = make_sinc_pulse(flip_angle=np.deg2rad(90), duration=3e-3, phase_offset=phase_exc,
                                   slice_thickness=5e-3, apodization=0.5, time_bw_product=4, system=system, return_gz=True)
    gz90_rephaser = make_trapezoid(channel='z', area=-gz90.area / 2, duration=1e-3, system=system)

    # Refocusing
    rf180, gz180, _ = make_sinc_pulse(flip_angle=np.deg2rad(refoc_angle_deg), duration=3e-3, phase_offset=phase_ref,
                                    slice_thickness=5e-3, apodization=0.5, time_bw_product=4, system=system, return_gz=True)
    
    adc = make_adc(num_samples=adc_samples, duration=3.2e-3, system=system, phase_offset=phase_exc)
    
    tau = te / 2
    rf90_dur = calc_duration(rf90, gz90)
    gz90_rephaser_dur = calc_duration(gz90_rephaser)
    rf180_dur = calc_duration(rf180, gz180)
    adc_dur = calc_duration(adc)
    raster = system.block_duration_raster
    
    seq.add_block(rf90, gz90)
    seq.add_block(gz90_rephaser)
    
    delay1 = tau - (rf90_dur / 2) - gz90_rephaser_dur - (rf180_dur / 2)
    if delay1 > 0: seq.add_block(make_delay(quantize(delay1, raster)))
    
    for _ in range(n_echo):
        seq.add_block(rf180, gz180)
        delay2 = tau - (rf180_dur / 2) - (adc_dur / 2)
        if delay2 > 0: seq.add_block(make_delay(quantize(delay2, raster)))
        seq.add_block(adc)
        delay3 = tau - (adc_dur / 2) - (rf180_dur / 2)
        if delay3 > 0: seq.add_block(make_delay(quantize(delay3, raster)))
    return seq

# SimData Setup (Single Voxel)
PD = torch.ones((1,))
T1 = torch.ones((1,)) * 1.0
T2 = torch.ones((1,)) * 0.1 # 100ms
T2dash = torch.ones((1,)) * 0.1
D = torch.zeros((1, 3)) 
B0 = torch.zeros((1,))
B1 = torch.ones((1, 1))
coil_sens = torch.ones((1, 1))
size = torch.tensor([0.005, 0.005, 0.005]) 
voxel_pos = torch.zeros((1, 3))
nyquist = torch.tensor([1, 1, 1])
dephasing_func = lambda b0, t: torch.zeros_like(b0)

data = mr0.SimData(PD, T1, T2, T2dash, D, B0, B1, coil_sens, size, voxel_pos, nyquist, dephasing_func)

# Run Simulations
TE = 10e-3
N_ECHO = 32
ADC_SAMPLES = 64

# 180 deg (Ideal)
seq_180 = build_cpmg_seq(te=TE, n_echo=N_ECHO, adc_samples=ADC_SAMPLES, refoc_angle_deg=180.0)
seq_180.write('output/seq/cpmg_180.seq')
raw_180 = simulate_bloch('output/seq/cpmg_180.seq', data, spin_count=500, print_progress=False)
sig_180 = np.abs(np.asarray(raw_180).reshape(N_ECHO, ADC_SAMPLES)).mean(axis=1)

# 60 deg (Stimulated)
seq_60 = build_cpmg_seq(te=TE, n_echo=N_ECHO, adc_samples=ADC_SAMPLES, refoc_angle_deg=60.0)
seq_60.write('output/seq/cpmg_60.seq')
raw_60 = simulate_bloch('output/seq/cpmg_60.seq', data, spin_count=500, print_progress=False)
sig_60 = np.abs(np.asarray(raw_60).reshape(N_ECHO, ADC_SAMPLES)).mean(axis=1)

# Plot
te_ms = (np.arange(N_ECHO) + 1) * TE * 1000
df_180 = pd.DataFrame({'TE': te_ms, 'Signal': sig_180, 'Label': '180 deg (Spin Echo)'})
df_60 = pd.DataFrame({'TE': te_ms, 'Signal': sig_60, 'Label': '60 deg (Stimulated Echo)'})
p = (
    ggplot(pd.concat([df_180, df_60]), aes(x='TE', y='Signal', color='Label'))
    + geom_line() + geom_point() + theme_bw()
    + labs(title='CPMG Signal Decay: Spin Echo vs Stimulated Echo')
)
print(p)
"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# --- Chapter 2 ---
text = """## 2. T1 Mapping: VFA, B1 Error, and AFI Correction

We simulate a realistic scenario for T1 mapping:
1.  **VFA (Variable Flip Angle):** SPGR sequences with varying angles.
2.  **Imperfections:** B1 error (actual FA is 85% of nominal) and Noise.
3.  **AFI (Actual Flip-angle Imaging):** Estimating the B1 map to correct T1.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code = """# 1. AFI Simulation (Estimate B1)
def build_afi_seq(flip_angle_deg, tr1=20e-3, tr2=100e-3, adc_samples=1, n_reps=200):
    seq = Sequence(system)
    rf, gz, _ = make_sinc_pulse(flip_angle=np.deg2rad(flip_angle_deg), duration=1e-3, time_bw_product=4, system=system, return_gz=True, slice_thickness=5e-3)
    gz_reph = make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
    adc = make_adc(num_samples=adc_samples, duration=3.2e-3, system=system)
    gz_spoil = make_trapezoid(channel='z', area=20/5e-3, duration=4e-3, system=system)
    
    rf_dur = calc_duration(rf, gz); gz_reph_dur = calc_duration(gz_reph); adc_dur = calc_duration(adc); spoil_dur = calc_duration(gz_spoil)
    min_dur = rf_dur + gz_reph_dur + adc_dur + spoil_dur
    delay1 = make_delay(quantize(tr1 - min_dur, system.block_duration_raster))
    delay2 = make_delay(quantize(tr2 - min_dur, system.block_duration_raster))
    
    for _ in range(n_reps):
        for d in [delay1, delay2]:
            seq.add_block(rf, gz); seq.add_block(gz_reph); seq.add_block(adc); seq.add_block(gz_spoil); seq.add_block(d)
    return seq

# Setup Data with B1 Error
TRUE_T1 = 1000.0; TRUE_B1 = 0.85
data_b1err = mr0.SimData(PD, torch.ones((1,))*1.0, T2, T2dash, D, B0, torch.ones((1,1))*TRUE_B1, coil_sens, size, voxel_pos, nyquist, dephasing_func)

# Run AFI
seq_afi = build_afi_seq(flip_angle_deg=60, tr1=20e-3, tr2=100e-3, n_reps=200, adc_samples=1)
seq_afi.write('output/seq/afi.seq')
raw_afi = simulate_bloch('output/seq/afi.seq', data_b1err, spin_count=1000, perfect_spoiling=False, print_progress=False)
sig_afi = np.abs(np.asarray(raw_afi).reshape(200, 2))
s1, s2 = np.mean(sig_afi[-10:, 0]), np.mean(sig_afi[-10:, 1])

# Estimate B1
afi_model = B1AFI(nom_fa_deg=60, tr1_ms=20, tr2_ms=100)
est_b1 = afi_model.fit(np.array([s1, s2]))['b1_raw']
print(f"True B1: {TRUE_B1}, Estimated B1: {est_b1:.4f}")

# 2. VFA Simulation & Correction
def build_spgr_seq(flip_angle_deg, tr=15e-3, adc_samples=1, n_reps=200):
    seq = Sequence(system)
    rf, gz, _ = make_sinc_pulse(flip_angle=np.deg2rad(flip_angle_deg), duration=1e-3, time_bw_product=4, system=system, return_gz=True, slice_thickness=5e-3)
    gz_reph = make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
    adc = make_adc(num_samples=adc_samples, duration=3.2e-3, system=system)
    gz_spoil = make_trapezoid(channel='z', area=10/5e-3, duration=2e-3, system=system)
    rf_dur = calc_duration(rf, gz); gz_reph_dur = calc_duration(gz_reph); adc_dur = calc_duration(adc); spoil_dur = calc_duration(gz_spoil)
    delay = make_delay(quantize(tr - (rf_dur + gz_reph_dur + adc_dur + spoil_dur), system.block_duration_raster))
    for _ in range(n_reps):
        seq.add_block(rf, gz); seq.add_block(gz_reph); seq.add_block(adc); seq.add_block(gz_spoil); seq.add_block(delay)
    return seq

flip_angles = [2, 5, 10, 15, 20, 30, 45, 60]
signals = []
for fa in flip_angles:
    seq = build_spgr_seq(fa, tr=15e-3, n_reps=200)
    seq.write(f'output/seq/spgr_{fa}.seq')
    raw = simulate_bloch(f'output/seq/spgr_{fa}.seq', data_b1err, spin_count=1000, perfect_spoiling=False, print_progress=False)
    signals.append(np.abs(np.asarray(raw).reshape(200, 1)[-1, 0]))

# Add Noise
np.random.seed(42)
sig_noisy = np.abs(np.array(signals) + np.random.normal(0, np.max(signals)/50, size=len(signals)))

# Fit (Uncorrected vs Corrected)
vfa_nocorr = T1VFA(flip_angle_deg=np.array(flip_angles), tr_ms=15.0, b1=1.0)
vfa_corr = T1VFA(flip_angle_deg=np.array(flip_angles), tr_ms=15.0, b1=est_b1)

t1_nocorr = vfa_nocorr.fit(sig_noisy)['t1_ms']
t1_corr = vfa_corr.fit(sig_noisy)['t1_ms']

print(f"True T1: {TRUE_T1} ms")
print(f"Uncorrected Fit (B1=1.0): {t1_nocorr:.1f} ms")
print(f"Corrected Fit (B1={est_b1:.2f}): {t1_corr:.1f} ms")
"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# --- Chapter 3 ---
text = """## 3. T2 Mapping & MWF Estimation

We simulate a voxel with two compartments:
*   Myelin Water (20%, T2=20ms)
*   IE Water (80%, T2=80ms)

We fit the data using Mono-exponential model and Multi-component (MWF) model.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code = """# Simulation: Two Compartments
def create_data(t2, fraction):
    PD_ = torch.ones((1,))*fraction; T2_ = torch.ones((1,))*t2/1000.0
    return mr0.SimData(PD_, T1, T2_, T2_, D, B0, B1, coil_sens, size, voxel_pos, nyquist, dephasing_func)

data_mw = create_data(20.0, 0.2); data_iew = create_data(80.0, 0.8)
seq_mwf = build_cpmg_seq(te=10e-3, n_echo=32, adc_samples=1)
seq_mwf.write('output/seq/mwf.seq')

raw_mw = simulate_bloch('output/seq/mwf.seq', data_mw, spin_count=500, print_progress=False)
raw_iew = simulate_bloch('output/seq/mwf.seq', data_iew, spin_count=500, print_progress=False)
sig_total = np.abs(np.asarray(raw_mw) + np.asarray(raw_iew)).reshape(32, 1).mean(axis=1)

# Add Noise
sig_noisy = np.abs(sig_total + np.random.normal(0, sig_total[0]/200, size=sig_total.shape))

# 1. Mono Fit
te_ms = (np.arange(32) + 1) * 10.0
t2_mono = T2Mono(te_ms).fit(sig_noisy)['t2_ms']

# 2. Multi Fit
res_mwf = T2MultiComponent(te_ms).fit(sig_noisy)
mwf_est = res_mwf['mwf']

print(f"Mono T2: {t2_mono:.1f} ms (Average)")
print(f"Estimated MWF: {mwf_est:.3f} (True: 0.200)")

# Plot Spectrum
df_spec = pd.DataFrame({'T2': res_mwf['t2_basis_ms'], 'Amplitude': res_mwf['weights']})
p = (ggplot(df_spec, aes(x='T2', y='Amplitude')) + geom_col(fill='purple') + scale_x_log10() + theme_bw() + labs(title=f'T2 Spectrum (MWF={mwf_est:.2f})'))
print(p)
"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# --- Chapter 4 ---
text = """## 4. T2* and Dephasing (FID vs Spin Echo)

Demonstration of reversible dephasing (Spin Echo) vs irreversible dephasing (T2*).
We simulate 5000 isochromats with random B0 offsets.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code = """# Data with B0 Inhomogeneity
SPIN_COUNT = 5000
B0_offsets = np.clip(np.random.standard_cauchy(SPIN_COUNT) * 10.0, -200, 200) # Lorentzian
data_t2s = mr0.SimData(torch.ones(SPIN_COUNT)/SPIN_COUNT, torch.ones(SPIN_COUNT)*1.0, torch.ones(SPIN_COUNT)*0.1, torch.ones(SPIN_COUNT)*0.1, 
                       torch.zeros(SPIN_COUNT), torch.tensor(B0_offsets, dtype=torch.float32), 
                       torch.ones((1, SPIN_COUNT)), torch.ones((1, SPIN_COUNT)), size, torch.zeros((SPIN_COUNT,3)), nyquist, lambda b0,t: 2*np.pi*b0*t)

# FID
seq_fid = Sequence(system)
rf90 = make_sinc_pulse(flip_angle=np.deg2rad(90), duration=1e-3, system=system, return_gz=False)
adc_fid = make_adc(num_samples=200, duration=100e-3, system=system)
seq_fid.add_block(rf90); seq_fid.add_block(adc_fid)
seq_fid.write('output/seq/fid.seq')
sig_fid = np.abs(np.asarray(simulate_bloch('output/seq/fid.seq', data_t2s, spin_count=1, print_progress=False)))

# Spin Echo (TE=50ms)
seq_se = Sequence(system)
rf180 = make_sinc_pulse(flip_angle=np.deg2rad(180), duration=1e-3, phase_offset=np.pi/2, system=system, return_gz=False)
adc_se = make_adc(num_samples=200, duration=40e-3, system=system)
rf90_d = calc_duration(rf90); rf180_d = calc_duration(rf180)
tau = 50e-3 / 2
seq_se.add_block(rf90)
seq_se.add_block(make_delay(quantize(tau - rf90_d/2 - rf180_d/2, system.block_duration_raster)))
seq_se.add_block(rf180)
seq_se.add_block(make_delay(quantize(tau - rf180_d/2 - 40e-3/2, system.block_duration_raster)))
seq_se.add_block(adc_se)
seq_se.write('output/seq/se.seq')
sig_se = np.abs(np.asarray(simulate_bloch('output/seq/se.seq', data_t2s, spin_count=1, print_progress=False)))

# Plot
df_fid = pd.DataFrame({'Time': np.linspace(0, 100, 200), 'Signal': sig_fid.flatten(), 'Type': 'FID (T2*)'})
df_se = pd.DataFrame({'Time': np.linspace(30, 70, 200), 'Signal': sig_se.flatten(), 'Type': 'Spin Echo'})
p = (ggplot(pd.concat([df_fid, df_se]), aes(x='Time', y='Signal', color='Type')) + geom_line() + theme_bw() + labs(title='Reversible Dephasing'))
print(p)
"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# --- Chapter 5 ---
text = """## 5. QSM Simulation

We create a 3D phantom (Sphere with $\chi=1$ppm), compute the dipole field, generate analytic GRE signals, and reconstruct the susceptibility map.
"""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code = """# 3D Phantom
N = 32; FOV = 0.2; dx = FOV/N
x = np.linspace(-FOV/2, FOV/2, N); X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
chi = np.zeros((N, N, N)); chi[X**2 + Y**2 + Z**2 <= 0.04**2] = 1.0

# Dipole Field
gamma = 42.58e6; B0_T = 3.0
D = kspace_kernel(np.array([FOV*1000]*3), (N, N, N))
DF = np.fft.ifftn(D * np.fft.fftn(chi)).real * (gamma * B0_T * 1e-6)

# Analytic Signal
TEs = [5e-3, 10e-3, 15e-3, 20e-3]
images = np.stack([np.exp(-te/0.1) * np.exp(-1j * 2 * np.pi * DF * te) for te in TEs], axis=-1)

# QSM Reconstruction
dPhi = np.angle(images[..., 1] * np.conj(images[..., 0]))
field_map = dPhi # Phase at delta_TE
qsm = QSMSplitBregman(sharp_filter=True, l1_regularized=True, lambda_l1=1e-3, pad_size=(4, 4, 4))
res = qsm.fit(phase=dPhi, mask=np.ones((N,N,N)), image_resolution_mm=np.array([dx*1000]*3))

# Compare Center Slice
sl = N//2
plt.figure(figsize=(10,3))
plt.subplot(131); plt.imshow(chi[:,:,sl], cmap='gray'); plt.title('True Chi')
plt.subplot(132); plt.imshow(DF[:,:,sl], cmap='jet'); plt.title('Field Map')
plt.subplot(133); plt.imshow(res['chi_sb'][:,:,sl] * (1.0/(gamma*B0_T*(TEs[1]-TEs[0])*1e-6)), cmap='gray'); plt.title('Recon Chi')
plt.show()
"""
nb['cells'].append(nbf.v4.new_code_cell(code))

# Write File
with open('notebooks/mrzero_tutorial.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Generated notebooks/mrzero_tutorial.ipynb")
