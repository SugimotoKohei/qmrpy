# qmrpy

[![PyPI](https://img.shields.io/pypi/v/qmrpy.svg)](https://pypi.org/project/qmrpy/)

qMRLab（MATLAB実装）の概念・モデルを **Python** へ段階的に移植するためのリポジトリです。

本プロジェクトは upstream の qMRLab（MIT License）に着想を得ており、モデル定義・検証方針は qMRLab を参照しつつ Python で再構成します。


## 開発（ローカル）

現時点では最小のパッケージ雛形のみです（今後、モデル実装を段階的に追加します）。

- `uv sync --extra viz`（可視化を含める）
- `uv sync --extra viz --extra dev`（pytest/ruff 等を含める）
- `uv run --locked -m pytest`

## パッケージ利用

`uv` を使う場合の導入例：

```bash
uv add qmrpy
```

## API規約（簡易）

- 物理量＋単位で命名（例：`t1_ms`, `t2_ms`, `flip_angle_deg`）。
- `forward(**params)` はシミュレーション出力を返す。
- `fit(signal, **kwargs)` は単一ボクセル向け推定で `dict` を返す。
- `fit_image(data, mask=None, **kwargs)` は画像/ボリューム向け推定で `dict` を返す。
  - `data` は `(..., n_obs)` の形、`mask` は空間形状と一致。
- 主要推定量は固定キー（例：`t1_ms`, `t2_ms`, `m0`）、補助量は意味が明確なキーで返す。
- 入力の shape 不一致や不正値は `ValueError` を投げる。
  - 例外として `fit_image` が 1D データを受け取る場合、`mask` は **禁止**（`ValueError`）。
- 関数API（`qmrpy.<func>`）は対応するモデルの `forward` / `fit` と同じ入出力規約に従う。

### 返却キー（T2系）

- `MonoT2.fit`: `m0`, `t2_ms`（`offset_term=True` なら `offset` も返す）
- `DecaesT2Map.fit`: `distribution`, `echotimes_ms`, `t2times_ms`, `alpha_deg`, `gdn`, `ggm`, `gva`, `fnr`, `snr`
  - 追加オプションで `mu`, `chi2factor`, `resnorm`, `decaycurve`, `decaybasis` が付く
- `DecaesT2Part.fit`: `sfr`, `sgm`, `mfr`, `mgm`
- `MultiComponentT2.fit`: `weights`, `t2_basis_ms`, `mwf`, `t2mw_ms`, `t2iew_ms`, `gmt2_ms`,
  `mw_weight`, `iew_weight`, `total_weight`, `resid_l2`

### 返却キー（B1 / Noise / QSM系）

- `B1Afi.fit_raw` / `B1Dam.fit_raw`: `b1_raw`, `spurious`
- `B1Afi.fit_image` / `B1Dam.fit_image`: `b1_raw`, `spurious`
- `MPPCA.fit`: `denoised`, `sigma`, `n_pars`
- `MPPCA.fit_image`: `denoised`, `sigma`, `n_pars`
- `QsmSplitBregman.fit`:
  - 常に `unwrapped_phase`, `mask_out`
  - `no_regularization=True` なら `nfm`
  - `l2_regularized=True` なら `chiL2`（必要なら `chiL2pcg`）
  - `l1_regularized=True` なら `chiSB`
- `qsm_split_bregman`: `chi`（再構成結果）
- `calc_chi_l2`: `chi_l2`, `chi_l2_pcg`

### 関数API（functional）

- `vfa_t1_forward`, `vfa_t1_fit_linear`
- `inversion_recovery_forward`, `inversion_recovery_fit`
- `mono_t2_forward`, `mono_t2_fit`
- `mwf_fit`
- `decaes_t2map_fit`, `decaes_t2map_spectrum`

### 最小利用例

```python
import numpy as np
from qmrpy.models.t1.vfa_t1 import VfaT1

model = VfaT1(
    tr_ms=15.0,
    flip_angle_deg=np.array([2, 5, 10, 15]),
)

signal = model.forward(m0=1.0, t1_ms=1200.0)
fit = model.fit_linear(signal)
print(fit["t1_ms"], fit["m0"])
```

関数API（オブジェクト不要）:

```python
import numpy as np
from qmrpy import vfa_t1_fit_linear

signal = np.array([0.02, 0.06, 0.12, 0.18], dtype=float)
fit = vfa_t1_fit_linear(
    signal,
    flip_angle_deg=np.array([2, 5, 10, 15]),
    tr_ms=15.0,
)
print(fit["t1_ms"], fit["m0"])
```

### QSM の最小利用例

```python
import numpy as np
from qmrpy.models.qsm import QsmSplitBregman

shape = (6, 6, 6)
phase = np.random.default_rng(0).normal(0, 1, size=shape)
mask = np.ones(shape, dtype=float)

qsm = QsmSplitBregman(
    sharp_filter=False,
    l1_regularized=True,
    l2_regularized=False,
    no_regularization=False,
    pad_size=(1, 1, 1),
)

out = qsm.fit(phase, mask, image_resolution_mm=[1.0, 1.0, 1.0])
print(out.keys())
```

## ライセンス

- `qmrpy` 本体：MIT（`LICENSE`）
- 参照元 `qMRLab/`：MIT（upstream、ローカル参照用）
- 翻訳・参考実装・vendor の詳細は `THIRD_PARTY_NOTICES.md` を参照

## 第三者由来コードの扱い

- `qMRLab`（MATLAB）および `DECAES.jl` の概念・アルゴリズムを翻訳/再構成しています。
- ライセンス表記・出自は `THIRD_PARTY_NOTICES.md` に集約しています。
