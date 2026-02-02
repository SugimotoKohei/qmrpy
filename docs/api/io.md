# I/O Utilities

Functions for saving and loading image data.

## TIFF I/O

::: qmrpy.io.save_tiff

::: qmrpy.io.load_tiff

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
