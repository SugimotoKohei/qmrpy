# T2 Mapping

qmrpy provides several T2 mapping methods:

- **MonoT2**: Simple mono-exponential decay
- **EpgT2**: EPG-corrected T2 with stimulated echo compensation
- **MultiComponentT2**: Multi-exponential T2 for myelin water fraction (MWF)
- **DecaesT2Map**: DECAES-style regularized NNLS fitting

## Mono-exponential T2

### Basic Usage

```python
import numpy as np
from qmrpy.models.t2 import MonoT2

# Echo times
te_ms = np.array([10, 20, 30, 40, 50])

# Create model
model = MonoT2(te_ms=te_ms)

# Fit
signal = np.array([100, 80, 64, 51, 41])
result = model.fit(signal)
print(f"T2 = {result['t2_ms']:.1f} ms")
```

### With Offset Term

For signals with non-zero baseline:

```python
result = model.fit(signal, offset_term=True)
print(f"T2 = {result['t2_ms']:.1f} ms, offset = {result['offset']:.1f}")
```

## EPG-corrected T2

Accounts for stimulated echoes in multi-echo spin-echo sequences:

```python
from qmrpy.models.t2 import EpgT2

# Create model
model = EpgT2(
    n_te=32,           # Number of echoes
    te_ms=10.0,        # Echo spacing
    t1_ms=1000.0,      # T1 (needed for EPG)
    alpha_deg=180.0,   # Refocusing flip angle
)

# Fit
result = model.fit(signal)
```

### B1 Correction

EPG fitting can incorporate B1 maps:

```python
# Single voxel with B1 correction
result = model.fit(signal, b1=0.95)

# Image fitting with B1 map
maps = model.fit_image(volume, b1_map=b1_map, n_jobs=-1)
```

## Multi-Component T2 (MWF)

For myelin water fraction estimation:

```python
from qmrpy.models.t2 import MultiComponentT2

# Create model with T2 basis
model = MultiComponentT2(
    te_ms=te_ms,
    t2_basis_ms=np.logspace(np.log10(10), np.log10(2000), 100),
    mw_range_ms=(10, 40),    # Myelin water T2 range
    iew_range_ms=(40, 200),  # Intra/extra-cellular water range
)

# Fit
result = model.fit(signal)
print(f"MWF = {result['mwf']:.2%}")
print(f"T2 myelin water = {result['t2mw_ms']:.1f} ms")
```

## DECAES T2 Distribution

Advanced T2 distribution fitting with regularization:

```python
from qmrpy.models.t2 import DecaesT2Map

model = DecaesT2Map(
    n_te=32,
    te_ms=10.0,
    n_t2=60,                    # Number of T2 bins
    t2_range_ms=(10.0, 2000.0),
    reg="lcurve",               # Regularization: lcurve, gcv, chi2
    t1_ms=1000.0,
)

# Fit returns (maps, distributions)
maps, dist = model.fit_image(volume, mask="otsu")
```

## Forward Models

```python
from qmrpy import mono_t2_forward, epg_t2_forward

# Mono-exponential
signal = mono_t2_forward(m0=100, t2_ms=50, te_ms=te_ms)

# EPG-corrected
signal = epg_t2_forward(
    m0=100, t2_ms=50, n_te=32, te_ms=10.0, t1_ms=1000.0
)
```

## API Reference

- [MonoT2](../api/t2.md#qmrpy.models.t2.MonoT2)
- [EpgT2](../api/t2.md#qmrpy.models.t2.EpgT2)
- [MultiComponentT2](../api/t2.md#qmrpy.models.t2.MultiComponentT2)
- [DecaesT2Map](../api/t2.md#qmrpy.models.t2.DecaesT2Map)
