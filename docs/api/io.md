# I/O Utilities

Functions for saving and loading image data. TIFF is available in the core
installation. NIfTI, DICOM, and BIDS helpers require the optional `io`
dependencies.

## TIFF I/O

::: qmrpy.io.save_tiff

::: qmrpy.io.load_tiff

## NIfTI I/O

::: qmrpy.io.save_nifti

::: qmrpy.io.load_nifti

::: qmrpy.io.save_nifti_map

## DICOM I/O

::: qmrpy.io.load_dicom_series

## BIDS I/O

::: qmrpy.io.load_bids_relaxometry

## Usage Examples

### Save T2 Map as TIFF

```python
import numpy as np
from qmrpy import save_tiff

# Save 2D map
t2_map = np.random.rand(256, 256).astype(np.float32) * 100
save_tiff("t2_map.tiff", t2_map)

# Save 3D volume (multi-page TIFF)
t2_volume = np.random.rand(10, 256, 256).astype(np.float32) * 100
save_tiff("t2_volume.tiff", t2_volume)

# Save with specific dtype
save_tiff("t2_map_uint16.tiff", t2_map, dtype="uint16")
```

### Load TIFF

```python
from qmrpy import load_tiff

# Load 2D or 3D TIFF
data = load_tiff("t2_map.tiff")
print(data.shape)  # (256, 256) or (10, 256, 256)
```

### Save a Fit Map as NIfTI

```python
from qmrpy.io import load_nifti, save_nifti_map
from qmrpy.models import T2Mono

data, affine, header = load_nifti("sub-01_echoes.nii.gz")
model = T2Mono(te_ms=[10, 20, 40, 80])
maps = model.fit_image(data, mask="otsu", n_jobs=-1)
save_nifti_map("sub-01_t2map.nii.gz", maps, "t2_ms", affine=affine, header=header)
```
