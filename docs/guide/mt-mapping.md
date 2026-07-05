# Magnetization Transfer Mapping

qmrpy provides two lightweight magnetization transfer helpers:

- **MTR**: ratio map from MT-off and MT-on images.
- **MTsat**: MT saturation map with spoiled-GRE T1/flip-angle/TR correction.

## MTR

```python
from qmrpy.models import MTR

model = MTR()
result = model.fit([1000, 800])
print(result["mtr"])          # 0.2
print(result["mtr_percent"])  # 20.0

maps = model.fit_image(image_data, mask="otsu", n_jobs=-1)
mtr_map = maps["mtr"]
```

`image_data` must have last dimension 2 as `[S0, Smt]`.

## MTsat

MTsat uses the MT-weighted signal, an `m0` map, and a `t1_ms` map. The model
predicts the spoiled-GRE signal without MT preparation from `m0`, `T1`,
flip angle, and TR, then estimates the additional MT saturation.

```python
from qmrpy.models import MTsat

model = MTsat(flip_angle_deg=6.0, tr_ms=25.0)
result = model.fit([120.0], m0=1000.0, t1_ms=1000.0)
print(result["mtsat"])

maps = model.fit_image(mt_weighted[..., None], m0=m0_map, t1_ms=t1_map, mask="otsu", n_jobs=-1)
mtsat_map = maps["mtsat"]
```

`mt_weighted[..., None]` must have last dimension 1 as `[Smt]`.

## API Reference

- [MTR](../api/mt.md#qmrpy.models.mt.MTR)
- [MTsat](../api/mt.md#qmrpy.models.mt.MTsat)
