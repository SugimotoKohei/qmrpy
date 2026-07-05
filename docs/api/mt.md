# Magnetization Transfer Models

MTR and MTsat mapping models.

## MTR

::: qmrpy.models.mt.MTR

## MTsat

::: qmrpy.models.mt.MTsat

## Usage

```python
from qmrpy.models import MTR, MTsat

mtr = MTR().fit([1000, 800])
print(mtr["mtr"])

mtsat_model = MTsat(flip_angle_deg=6.0, tr_ms=25.0)
mtsat = mtsat_model.fit([120.0], m0=1000.0, t1_ms=1000.0)
print(mtsat["mtsat"])
```
