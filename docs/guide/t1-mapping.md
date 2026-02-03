# T1 Mapping

qmrpy provides two T1 mapping methods:

- **VFA (Variable Flip Angle)**: Fast T1 mapping using FLASH sequence
- **Inversion Recovery**: Gold standard T1 measurement

## VFA T1 Mapping

### Basic Usage

```python
import numpy as np
from qmrpy.models.t1 import VFAT1

# Acquisition parameters
flip_angles = np.array([3, 15])  # degrees
tr_ms = 20.0  # milliseconds

# Create model
model = VFAT1(flip_angle_deg=flip_angles, tr_ms=tr_ms)

# Fit single voxel
signal = np.array([50, 200])
result = model.fit(signal)
print(f"T1 = {result['t1_ms']:.0f} ms")
```

### B1 Correction

VFA requires B1 correction for accurate results:

```python
# With B1 map (per-voxel)
model = VFAT1(flip_angle_deg=flip_angles, tr_ms=tr_ms, b1=0.95)

# Or pass B1 during fit
result = model.fit(signal)  # Uses model's b1
```

### Image Fitting

```python
# Load your data (shape: x, y, z, n_flip_angles)
vfa_data = np.load("vfa_data.npy")

# Fit with Otsu masking and parallel processing
maps = model.fit_image(vfa_data, mask="otsu", n_jobs=-1, verbose=True)
t1_map = maps["t1_ms"]
m0_map = maps["m0"]
```

## Inversion Recovery

### Basic Usage

```python
from qmrpy.models.t1 import InversionRecovery

# Inversion times
ti_ms = np.array([50, 100, 200, 400, 800, 1600, 3200])

# Create model
model = InversionRecovery(ti_ms=ti_ms)

# Fit (magnitude data)
signal = np.abs(np.array([-95, -85, -60, -10, 50, 80, 95]))
result = model.fit(signal, method="magnitude")
print(f"T1 = {result['t1_ms']:.0f} ms")
```

### Complex vs Magnitude Fitting

```python
# For complex/signed data
result = model.fit(signal, method="complex")

# For magnitude data (handles sign ambiguity)
result = model.fit(np.abs(signal), method="magnitude")
```

## Forward Models

Generate synthetic signals:

```python
from qmrpy import vfa_t1_forward, inversion_recovery_forward

# VFA signal
signal = vfa_t1_forward(
    m0=1000, t1_ms=1000, 
    flip_angle_deg=flip_angles, tr_ms=tr_ms
)

# Inversion recovery signal
signal = inversion_recovery_forward(
    t1_ms=1000, ra=-1000, rb=2000, ti_ms=ti_ms
)
```

## API Reference

- [VFAT1](../api/t1.md#qmrpy.models.t1.VFAT1)
- [InversionRecovery](../api/t1.md#qmrpy.models.t1.InversionRecovery)
