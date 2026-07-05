# MR Fingerprinting Models

Dictionary-based simultaneous T1-T2 mapping.

## MRFDictionary

::: qmrpy.models.mrf.MRFDictionary

## Usage

```python
from qmrpy.models import MRFDictionary

model = MRFDictionary(
    flip_angle_deg=[5, 25, 10, 35, 15, 30],
    tr_ms=[12, 14, 11, 16, 13, 15],
    te_ms=[3, 6, 4, 8, 5, 7],
)

signal = model.forward(m0=1000, t1_ms=1000, t2_ms=80)
fit = model.fit(signal, t1_grid_ms=[800, 1000, 1200], t2_grid_ms=[60, 80, 100])
print(fit["t1_ms"], fit["t2_ms"])
```
