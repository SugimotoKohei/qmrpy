# B1 Mapping

qmrpy provides two B1+ mapping methods:

- **DAM (Double Angle Method)**: Simple and fast
- **AFI (Actual Flip Angle Imaging)**: More robust

## Double Angle Method (DAM)

### Basic Usage

```python
import numpy as np
from qmrpy.models.b1 import B1Dam

# Nominal flip angle
model = B1Dam(alpha_deg=60.0)

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
b1_map = maps["b1_raw"]
spurious = maps["spurious"]  # Quality flag
```

## Actual Flip Angle Imaging (AFI)

### Basic Usage

```python
from qmrpy.models.b1 import B1Afi

# AFI parameters
model = B1Afi(
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
b1_map = maps["b1_raw"]
```

## Using B1 Maps for Correction

### VFA T1 with B1 Correction

```python
from qmrpy.models.t1 import VfaT1

model = VfaT1(flip_angle_deg=[3, 15], tr_ms=20.0, b1=b1_map)
t1_maps = model.fit_image(vfa_data)
```

### EPG T2 with B1 Correction

```python
from qmrpy.models.t2 import EpgT2

model = EpgT2(n_te=32, te_ms=10.0)
t2_maps = model.fit_image(t2_data, b1_map=b1_map)
```

## Quality Filtering

Both methods return a `spurious` map indicating unreliable fits:

```python
maps = model.fit_image(data)

# Filter unreliable B1 values
b1_clean = maps["b1_raw"].copy()
b1_clean[maps["spurious"] > 0] = np.nan
```

## API Reference

- [B1Dam](../api/b1.md#qmrpy.models.b1.B1Dam)
- [B1Afi](../api/b1.md#qmrpy.models.b1.B1Afi)
