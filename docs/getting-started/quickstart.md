# Quick Start

This guide covers the basic workflow for fitting qMRI models.

## Basic Workflow

All models follow a consistent pattern:

1. Create a model with acquisition parameters
2. Call `fit()` for single voxel or `fit_image()` for volumes

## Example: Mono-exponential T2

```python
import numpy as np
from qmrpy.models.t2 import T2Mono

# Echo times in milliseconds
te_ms = np.array([10, 20, 30, 40, 50])

# Create model
model = T2Mono(te_ms=te_ms)

# Single voxel fit
signal = np.array([100, 80, 64, 51, 41])
result = model.fit(signal)
print(f"T2 = {result['t2_ms']:.1f} ms, M0 = {result['m0']:.1f}")
```

## Image Fitting

For multi-dimensional data, use `fit_image()`:

```python
# 3D volume with echoes as last dimension
volume = np.random.rand(64, 64, 10, 5) * 100  # (x, y, z, n_echoes)

# Fit with automatic Otsu masking
maps = model.fit_image(volume, mask="otsu")
t2_map = maps["t2_ms"]  # Shape: (64, 64, 10)
# 追加情報は maps.quality / maps.diagnostics で参照可能
```

## Parallel Processing

Enable parallel fitting with `n_jobs`:

```python
# Use all CPU cores
maps = model.fit_image(volume, mask="otsu", n_jobs=-1)
```

## Progress Bar

Show progress during fitting:

```python
import logging
logging.basicConfig(level=logging.INFO)

maps = model.fit_image(volume, mask="otsu", n_jobs=-1, verbose=True)
# Output: T2Mono: 4096 voxels, n_jobs=-1, shape=(64, 64, 10)
#         T2Mono: 100%|██████| 4096/4096 [00:05<00:00, 800voxel/s]
```

## Functional API

For quick one-off fits:

```python
from qmrpy import fit_t2_mono

result = fit_t2_mono(signal, te_ms=te_ms)
```

## Next Steps

- [T1 Mapping Guide](../guide/t1-mapping.md)
- [T2 Mapping Guide](../guide/t2-mapping.md)
- [API Reference](../api/index.md)
