# qmrpy

[![PyPI](https://img.shields.io/pypi/v/qmrpy.svg)](https://pypi.org/project/qmrpy/)

Python toolkit for quantitative MRI (qMRI) modeling, fitting, and simulation.

## Statement of need

qMRLab (MATLAB) and DECAES (Julia) provide widely used qMRI reference implementations, but integration
into Python workflows can be frictional. qmrpy reimplements key models in Python with a unified API,
explicit parameter naming, and reproducible tests so researchers can run model fitting and comparisons
inside a single Python environment.

## Scope

- Models: T1, T2, B1, QSM, noise, and simulation utilities.
- Interfaces: object-oriented models plus functional wrappers.
- Verification: parity checks for DECAES components and a broad test suite.

## Installation

```bash
pip install qmrpy
```

If you use `uv`:

```bash
uv add qmrpy
```

## Quickstart

```python
import numpy as np
from qmrpy.models.t1.vfa_t1 import VfaT1

model = VfaT1(tr_ms=15.0, flip_angle_deg=np.array([2, 5, 10, 15]))

signal = model.forward(m0=1.0, t1_ms=1200.0)
fit = model.fit(signal)
print(fit["t1_ms"], fit["m0"])
```

Functional API (no object):

```python
import numpy as np
from qmrpy import vfa_t1_fit

signal = np.array([0.02, 0.06, 0.12, 0.18], dtype=float)
fit = vfa_t1_fit(signal, flip_angle_deg=np.array([2, 5, 10, 15]), tr_ms=15.0)
print(fit["t1_ms"], fit["m0"])
```

EPG-corrected T2 (multi-echo spin-echo):

```python
import numpy as np
from qmrpy.models.t2 import EpgT2

model = EpgT2(n_te=32, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)
signal = model.forward(m0=1.0, t2_ms=80.0)
fit = model.fit(signal)
print(fit["t2_ms"], fit["m0"])
```

Optional B1 correction:
`forward(..., b1=0.9)`, `fit(..., b1=0.9)`, or `fit_image(..., b1_map=...)`.

Functional API (EPG-corrected T2):

```python
import numpy as np
from qmrpy import epg_t2_fit, epg_t2_forward

signal = epg_t2_forward(
    m0=1.0, t2_ms=80.0, n_te=32, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0
)
fit = epg_t2_fit(signal, n_te=32, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)
print(fit["t2_ms"], fit["m0"])
```

B1 mapping integration (example):

```python
import numpy as np
from qmrpy.models.b1 import B1Dam
from qmrpy.models.t2 import EpgT2

b1_model = B1Dam(alpha_deg=60.0)
sig_b1 = b1_model.forward(m0=1.0, b1=0.9)
b1_fit = b1_model.fit(sig_b1)

t2_model = EpgT2(n_te=16, te_ms=10.0, t1_ms=1000.0, alpha_deg=180.0)
sig_t2 = t2_model.forward(m0=1.0, t2_ms=80.0, b1=b1_fit["b1_raw"])
t2_fit = t2_model.fit(sig_t2, b1=b1_fit["b1_raw"])
print(t2_fit["t2_ms"])
```

Use `b1_map` for voxel-wise correction:
`t2_model.fit_image(data, b1_map=..., mask=...)`.

## API overview

- Names use physical quantity + unit (e.g., `t1_ms`, `t2_ms`, `flip_angle_deg`).
- `forward(**params)` returns simulated signal(s).
- `fit(signal, **kwargs)` returns a `dict` for a single voxel.
- `fit_image(data, mask=None, **kwargs)` returns a `dict` for images/volumes.
  - `data` shape is `(..., n_obs)` and `mask` matches spatial shape.
- Primary estimates use fixed keys (e.g., `t1_ms`, `t2_ms`, `m0`); auxiliaries use `snake_case`.

Model modules:

- `qmrpy.models.t1`
- `qmrpy.models.t2`
- `qmrpy.models.b1`
- `qmrpy.models.qsm`
- `qmrpy.models.noise`
- `qmrpy.sim`

## Verification highlights

### DECAES parity (reference CSV)

Errors against reference CSV in `tests/data` (regenerate with `uv run scripts/summarize_parity.py --no-qmrlab`).

| reg | abs(alpha-alpha_ref) [deg] | abs(mu-mu_ref) | abs(chi2factor-chi2_ref) | max abs(dist-dist_ref) |
|---|---|---|---|---|
| none | 0 | N/A | N/A | 2.22045e-15 |
| gcv | 1.99e-10 | 4.79026e-08 | N/A | 1.12119e-06 |
| lcurve | 1.99e-10 | 0 | 1.04894e-11 | 3.29564e-12 |
| chi2 | 1.99e-10 | 5.67276e-14 | 1.28786e-14 | 4.82459e-12 |
| mdp | 1.99e-10 | 5.66938e-14 | 1.69642e-13 | 4.91318e-12 |

### qMRLab parity (optional)

Requires Octave and a qMRLab checkout. The script compares qMRLab-generated signals against qmrpy fits.

```bash
QMRLAB_PATH=/path/to/qMRLab uv run scripts/verify_parity.py --model inversion_recovery
QMRLAB_PATH=/path/to/qMRLab uv run scripts/verify_qmrlab_mwf.py
QMRLAB_PATH=/path/to/qMRLab uv run scripts/sweep_qmrlab_mwf.py
```

Inversion Recovery (Barral, rdNLS):

| metric | abs diff |
|---|---|
| T1 (ms) | 0 |

MWF example:

| metric | abs diff |
|---|---|
| MWF (%) | 0.105 |
| T2MW (ms) | 0.127 |
| T2IEW (ms) | 0.045 |

## Tests

Run the test suite:

```bash
uv run --locked -m pytest
```

Generate a test matrix table (output under `output/`, not tracked):

```bash
uv run scripts/summarize_tests.py
```

## Development

```bash
uv sync --extra viz
uv sync --extra viz --extra dev
```

## Citation

For JOSS submission, see `paper.md` and `paper.bib`. A Zenodo archive DOI will be added before release.

## License

- `qmrpy` core: MIT (`LICENSE`)
- Third-party notices: `THIRD_PARTY_NOTICES.md`
