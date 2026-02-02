# qmrpy

**Quantitative MRI modeling library for Python**

qmrpy provides tools for fitting, simulating, and analyzing quantitative MRI data including T1, T2, B1 mapping, and QSM.

## Features

- 🧲 **T1 Mapping**: VFA, Inversion Recovery
- 🧲 **T2 Mapping**: Mono-exponential, EPG-corrected, Multi-component (MWF), DECAES
- 📡 **B1 Mapping**: DAM, AFI methods
- 🔬 **Simulation**: Bloch equation, EPG, phantoms
- ⚡ **Performance**: Parallel fitting with joblib, progress bars with tqdm
- 🎯 **Masking**: Built-in Otsu thresholding

## Quick Example

```python
import numpy as np
from qmrpy.models.t2 import MonoT2

# Setup
te_ms = np.array([10, 20, 30, 40, 50])
model = MonoT2(te_ms=te_ms)

# Fit single voxel
signal = np.array([100, 80, 64, 51, 41])
result = model.fit(signal)
print(f"T2 = {result['t2_ms']:.1f} ms")

# Fit 3D volume with progress bar
volume = np.random.rand(64, 64, 10, 5) * 100
maps = model.fit_image(volume, mask="otsu", n_jobs=-1, verbose=True)
```

## Installation

```bash
pip install qmrpy
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](guide/t1-mapping.md)
- [API Reference](api/index.md)

## License

BSD-2-Clause
