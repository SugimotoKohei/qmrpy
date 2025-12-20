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

### 最小利用例

```python
import numpy as np
from qmrpy.models.t1.vfa_t1 import VfaT1

model = VfaT1(
    tr_s=0.015,
    flip_angles_deg=np.array([2, 5, 10, 15]),
)

signal = model.forward(m0=1.0, t1_s=1.2)
fit = model.fit_linear(signal)
print(fit["t1_s"], fit["m0"])
```

## ライセンス

- `qmrpy` 本体：MIT（`LICENSE`）
- 参照元 `qMRLab/`：MIT（upstream、ローカル参照用）
- 翻訳・参考実装・vendor の詳細は `THIRD_PARTY_NOTICES.md` を参照

## 第三者由来コードの扱い

- `qMRLab`（MATLAB）および `DECAES.jl` の概念・アルゴリズムを翻訳/再構成しています。
- `epgpy` は `src/epgpy/` に vendor しています。
- ライセンス表記・出自は `THIRD_PARTY_NOTICES.md` に集約しています。
