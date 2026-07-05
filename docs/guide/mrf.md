# MR Fingerprinting

`MRFDictionary` provides a minimal dictionary-based simultaneous T1-T2 mapping
workflow. The current implementation uses the qmrpy EPG state engine to simulate
a spoiled FISP-like transient with variable flip angles and TRs.

```python
from qmrpy.models import MRFDictionary

model = MRFDictionary(
    flip_angle_deg=[5, 25, 10, 35, 15, 30],
    tr_ms=[12, 14, 11, 16, 13, 15],
    te_ms=[3, 6, 4, 8, 5, 7],
)

signal = model.forward(m0=1000, t1_ms=1000, t2_ms=80)
fit = model.fit(
    signal,
    t1_grid_ms=[800, 1000, 1200],
    t2_grid_ms=[60, 80, 100],
)
print(fit["t1_ms"], fit["t2_ms"])
print(fit.diagnostics["correlation"])
```

For image data, the last dimension is the MRF time axis:

```python
maps = model.fit_image(
    image_data,
    t1_grid_ms=[800, 1000, 1200],
    t2_grid_ms=[60, 80, 100],
    mask="otsu",
    n_jobs=-1,
)
```

This first implementation intentionally keeps the sequence model compact and
deterministic. It is suitable for synthetic validation and small dictionaries;
large protocol optimization, off-resonance modeling, and scanner-specific
dictionary compression are out of scope for this phase.

## API Reference

- [MRFDictionary](../api/mrf.md#qmrpy.models.mrf.MRFDictionary)
