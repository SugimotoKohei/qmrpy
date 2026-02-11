# B1 Mapping

qmrpy provides two B1+ mapping methods:

- **DAM (Double Angle Method)**: Simple and fast
- **AFI (Actual Flip Angle Imaging)**: More robust
- **Bloch-Siegert**: Phase-based B1 estimation

## Double Angle Method (DAM)

### Basic Usage

```python
import numpy as np
from qmrpy.models.b1 import B1DAM

# Nominal flip angle
model = B1DAM(alpha_deg=60.0)

# Signal at alpha and 2*alpha
signal = np.array([80, 140])  # S(60°), S(120°)

# Fit
result = model.fit(signal)
print(f"B1 = {result['b1_raw']:.2f}")  # 1.0 = nominal
```

### Image Fitting

```python
# Load DAM data (shape: x, y, z, 2)
dam_data = np.load("dam_data.npy")

# Fit (vectorized, very fast)
maps = model.fit_image(dam_data, mask="otsu")
b1_map = maps["params"]["b1_raw"]
spurious = maps["diagnostics"]["spurious"]  # Quality flag
```

## Actual Flip Angle Imaging (AFI)

### Basic Usage

```python
from qmrpy.models.b1 import B1AFI

# AFI parameters
model = B1AFI(
    nom_fa_deg=60.0,  # Nominal flip angle
    tr1_ms=20.0,      # Short TR
    tr2_ms=100.0,     # Long TR (TR ratio typically 5:1)
)

# Signal from two TRs
signal = np.array([80, 90])  # S(TR1), S(TR2)

# Fit
result = model.fit(signal)
print(f"B1 = {result['b1_raw']:.2f}")
```

### Image Fitting

```python
# Load AFI data (shape: x, y, z, 2)
afi_data = np.load("afi_data.npy")

# Fit
maps = model.fit_image(afi_data, mask="otsu")
b1_map = maps["params"]["b1_raw"]
```

## Bloch-Siegert

```python
from qmrpy.models.b1 import B1BlochSiegert

model = B1BlochSiegert(k_bs_rad_per_b1sq=2.0)

# Two phase/complex acquisitions with opposite off-resonance pulses
signal = np.array([s_plus, s_minus])
result = model.fit(signal)
print(f"B1 = {result['b1_raw']:.2f}")
```

## Using B1 Maps for Correction

### VFA T1 with B1 Correction

```python
from qmrpy.models.t1 import T1VFA

model = T1VFA(flip_angle_deg=[3, 15], tr_ms=20.0, b1=b1_map)
t1_maps = model.fit_image(vfa_data)
```

### EPG T2 with B1 Correction

```python
from qmrpy.models.t2 import T2EPG

model = T2EPG(n_te=32, te_ms=10.0)
t2_maps = model.fit_image(t2_data, b1_map=b1_map)
```

## B0 Mapping (for T2* support)

```python
from qmrpy.models.b0 import B0DualEcho

b0_model = B0DualEcho(te1_ms=4.0, te2_ms=6.0)
b0_maps = b0_model.fit_image(dual_echo_complex)
```

## Quality Filtering

Both methods return a `spurious` map indicating unreliable fits:

```python
maps = model.fit_image(data)

# Filter unreliable B1 values
b1_clean = maps["params"]["b1_raw"].copy()
b1_clean[maps["diagnostics"]["spurious"] > 0] = np.nan
```

## API Reference

- [B1DAM](../api/b1.md#qmrpy.models.b1.B1DAM)
- [B1AFI](../api/b1.md#qmrpy.models.b1.B1AFI)
- [B1BlochSiegert](../api/b1.md#qmrpy.models.b1.B1BlochSiegert)
- [B0DualEcho](../api/b0.md#qmrpy.models.b0.B0DualEcho)
