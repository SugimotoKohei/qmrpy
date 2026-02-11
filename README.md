<p align="center">
  <img src="https://raw.githubusercontent.com/SugimotoKohei/qmrpy/main/docs/assets/qmrpy-icon.png" alt="qmrpy logo" width="160">
</p>

# qmrpy

[![PyPI](https://img.shields.io/pypi/v/qmrpy.svg)](https://pypi.org/project/qmrpy/)

Python toolkit for quantitative MRI (qMRI) modeling, fitting, and simulation.

## Installation

```bash
pip install qmrpy
# or
uv add qmrpy
```

## Quickstart

```python
import numpy as np
from qmrpy.models import T2Mono

# Define model
model = T2Mono(te_ms=[10, 20, 40, 80])

# Simulate signal
signal = model.forward(m0=1000, t2_ms=80)

# Fit single voxel
fit = model.fit(signal)
print(fit)  # {'m0': 1000.0, 't2_ms': 80.0}

# Fit image with auto-masking and parallel processing
result = model.fit_image(image_data, mask="otsu", n_jobs=-1)
```

## Features

- **Models**: T1 (VFA, IR, DESPOT1-HIFI, T1MP2RAGE), T2/T2* (mono-exp, EPG, EMC, R2*), B0/B1, QSM, denoising
- **Parallel fitting**: `n_jobs=-1` for multi-core acceleration
- **Auto-masking**: `mask="otsu"` for automatic thresholding
- **I/O**: `save_tiff()` / `load_tiff()` for uncompressed TIFF export
- **Validation suite**: reproducible cross-domain checks for `T1/T2/B1/QSM/Simulation`

## Validation (JOSS-friendly, external dependency free)

Run the core validation suite:

```bash
uv run --locked -- python scripts/summarize_parity.py \
  --suite core \
  --formats csv,markdown,json \
  --config configs/exp/validation_core.toml \
  --out-dir output/reports/parity_summary
```

Key outputs:

- `output/reports/parity_summary/core_validation.csv`
- `output/reports/parity_summary/core_validation_metrics.csv`
- `output/reports/parity_summary/summary.md`
- `output/reports/parity_summary/summary.json`

## API

```python
from qmrpy.models import T2Mono, T2EPG, T1VFA, T1InversionRecovery
from qmrpy import save_tiff, load_tiff

# All models follow the same pattern:
model = Model(acquisition_params)
signal = model.forward(**tissue_params)
fit = model.fit(signal)
result = model.fit_image(image, mask="otsu", n_jobs=-1)
```

## License

MIT

---

# qmrpy（日本語）

定量MRI（qMRI）のモデリング・フィッティング・シミュレーション用Pythonツールキット。

## インストール

```bash
pip install qmrpy
# または
uv add qmrpy
```

## クイックスタート

```python
import numpy as np
from qmrpy.models import T2Mono

# モデル定義
model = T2Mono(te_ms=[10, 20, 40, 80])

# 信号シミュレーション
signal = model.forward(m0=1000, t2_ms=80)

# 単一ボクセルのフィッティング
fit = model.fit(signal)
print(fit)  # {'m0': 1000.0, 't2_ms': 80.0}

# 画像フィッティング（自動マスク＋並列処理）
result = model.fit_image(image_data, mask="otsu", n_jobs=-1)
```

## 主な機能

- **モデル**: T1（VFA, IR, DESPOT1-HIFI, T1MP2RAGE）、T2/T2*（単指数、EPG、EMC、R2*）、B0/B1、QSM、ノイズ除去
- **並列フィッティング**: `n_jobs=-1`でマルチコア高速化
- **自動マスク**: `mask="otsu"`でOtsu二値化
- **I/O**: `save_tiff()` / `load_tiff()`で非圧縮TIFF保存
- **検証スイート**: `T1/T2/B1/QSM/Simulation` を横断した再現可能な検証

## 検証実行（JOSS向け・外部依存なし）

```bash
uv run --locked -- python scripts/summarize_parity.py \
  --suite core \
  --formats csv,markdown,json \
  --config configs/exp/validation_core.toml \
  --out-dir output/reports/parity_summary
```

主要な成果物:

- `output/reports/parity_summary/core_validation.csv`
- `output/reports/parity_summary/core_validation_metrics.csv`
- `output/reports/parity_summary/summary.md`
- `output/reports/parity_summary/summary.json`

## ライセンス

MIT
