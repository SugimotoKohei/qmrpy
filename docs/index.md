# qmrpy

**Quantitative MRI modeling library for Python**

qmrpy provides tools for fitting, simulating, and analyzing quantitative MRI data including T1, T2, B1 mapping, and QSM.

## Features

- **T1 Mapping**: VFA, inversion recovery, DESPOT1-HIFI, MP2RAGE, T1rho
- **Magnetization Transfer**: MTR and MTsat
- **MR Fingerprinting**: dictionary-based simultaneous T1-T2 matching
- **T2/T2* Mapping**: mono-exponential, EPG/EMC, water/fat, MWF, R2*, ESTATICS
- **B1/B0 Mapping**: DAM, AFI, Bloch-Siegert, dual/multi-echo B0
- **QSM and denoising**: SHARP, split-Bregman QSM, MPPCA
- **Real-data I/O**: TIFF plus optional NIfTI, DICOM series, and minimal qMRI-BIDS helpers
- **CLI and validation**: `qmrpy fit`, `qmrpy info`, `qmrpy validate`
- **Performance**: parallel fitting, progress bars, Otsu auto-masking

## Quick Example

```python
import numpy as np
from qmrpy.models.t2 import T2Mono

# Setup
te_ms = np.array([10, 20, 30, 40, 50])
model = T2Mono(te_ms=te_ms)

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
pip install "qmrpy[io]"  # optional NIfTI/DICOM/BIDS helpers
```

## Documentation

- [Getting Started](getting-started/installation.md)
- [User Guide](guide/t1-mapping.md)
- [API Reference](api/index.md)

## License

MIT License
