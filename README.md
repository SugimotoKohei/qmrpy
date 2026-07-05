<p align="center">
  <img src="https://raw.githubusercontent.com/SugimotoKohei/qmrpy/main/docs/assets/qmrpy-icon.png?v=20260213" alt="qmrpy logo" width="160">
</p>

# qmrpy

[![PyPI](https://img.shields.io/pypi/v/qmrpy.svg)](https://pypi.org/project/qmrpy/)

Python toolkit for quantitative MRI (qMRI) modeling, fitting, and simulation.

## Installation

```bash
pip install qmrpy
# Optional real-data I/O helpers
pip install "qmrpy[io]"
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
print(fit["t2_ms"])            # params access (dict-like)
print(fit.quality["rmse"])     # quality metadata
print(fit.diagnostics)         # diagnostics metadata

# Fit image with auto-masking and parallel processing
result = model.fit_image(image_data, mask="otsu", n_jobs=-1)
```

## Features

- **Models**: T1 (VFA, IR, DESPOT1-HIFI, T1MP2RAGE), T1rho, MTR/MTsat, MRF, T2/T2* (mono-exp, EPG, EMC, water/fat, R2*), B0/B1, QSM, denoising
- **Parallel fitting**: `n_jobs=-1` for multi-core acceleration
- **Auto-masking**: `mask="otsu"` for automatic thresholding
- **I/O**: TIFF in core; optional NIfTI, DICOM, and BIDS helpers for real-data workflows
- **Validation suite**: reproducible cross-domain checks for `T1/T2/B1/QSM/Simulation`

## Real-data I/O

```python
from qmrpy.io import load_nifti, save_nifti_map
from qmrpy.models import T2Mono

data, affine, header = load_nifti("sub-01_echoes.nii.gz")
model = T2Mono(te_ms=[10, 20, 40, 80])
maps = model.fit_image(data, mask="otsu", n_jobs=-1)
save_nifti_map("sub-01_t2map.nii.gz", maps, "t2_ms", affine=affine, header=header)
```

`load_dicom_series()` returns sorted 3D or 4D arrays plus TE/TR/FA/TI metadata.
`load_bids_relaxometry()` reads minimal qMRI-BIDS NIfTI + JSON sidecar inputs and
normalizes timing fields to `_ms` keys.

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
# 実データ I/O の optional helper
pip install "qmrpy[io]"
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
print(fit["t2_ms"])            # params への辞書互換アクセス
print(fit.quality["rmse"])     # quality メタデータ
print(fit.diagnostics)         # diagnostics メタデータ

# 画像フィッティング（自動マスク＋並列処理）
result = model.fit_image(image_data, mask="otsu", n_jobs=-1)
```

## 主な機能

- **モデル**: T1（VFA, IR, DESPOT1-HIFI, T1MP2RAGE）、T1rho、MTR/MTsat、MRF、T2/T2*（単指数、EPG、EMC、水/脂肪、R2*）、B0/B1、QSM、ノイズ除去
- **並列フィッティング**: `n_jobs=-1`でマルチコア高速化
- **自動マスク**: `mask="otsu"`でOtsu二値化
- **I/O**: コアのTIFF入出力に加え、optionalでNIfTI/DICOM/BIDSの実データ入出力に対応
- **検証スイート**: `T1/T2/B1/QSM/Simulation` を横断した再現可能な検証

## 実データ I/O

```python
from qmrpy.io import load_nifti, save_nifti_map
from qmrpy.models import T2Mono

data, affine, header = load_nifti("sub-01_echoes.nii.gz")
model = T2Mono(te_ms=[10, 20, 40, 80])
maps = model.fit_image(data, mask="otsu", n_jobs=-1)
save_nifti_map("sub-01_t2map.nii.gz", maps, "t2_ms", affine=affine, header=header)
```

`load_dicom_series()` はソート済みの3D/4D配列とTE/TR/FA/TIメタデータを返します。
`load_bids_relaxometry()` は最小限の qMRI-BIDS NIfTI + JSON sidecar を読み込み、
時間フィールドを `_ms` キーへ正規化します。

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
