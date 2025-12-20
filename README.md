# qmrpy

[![PyPI](https://img.shields.io/pypi/v/qmrpy.svg)](https://pypi.org/project/qmrpy/)

qMRLab（MATLAB実装）の概念・モデルを **Python** へ段階的に移植するためのリポジトリです。

本プロジェクトは upstream の qMRLab（MIT License）に着想を得ており、モデル定義・検証方針は qMRLab を参照しつつ Python で再構成します。

## 方針（重要）

 - 出力（実験結果やログ）は原則 `output/` に出し、Git 管理しません。

## ディレクトリ構成（予定）

```text
.
├─ .github/                  # CI（pytest等）
│  └─ workflows/
├─ qMRLab/                   # 参照用: upstreamのMATLAB実装（ローカル配置・Git管理外）
├─ src/                      # Python実装（import対象）
│  └─ qmrpy/                  # 配布対象パッケージ
├─ tests/                    # pytest
├─ scripts/                  # 実験・実行入口（薄く保つ）
├─ configs/                  # 実験設定（再現性の入力）
├─ notebooks/                # 探索・可視化（任意）
└─ output/                   # 実験結果（Git外）
```

## 代表コマンド（後で有効化）

研究計画・記録は非公開で管理しています。

- `uv sync --locked --extra viz --extra dev`
- `uv run scripts/run_experiment.py --config configs/exp/mono_t2_baseline.toml`

### vfa_t1 比較run（おすすめ）

まずは以下の3条件を回して比較します（ノイズモデル/B1ばらつき/外れ値耐性の差を見る）：

- `uv run --locked scripts/run_experiment.py --config configs/exp/vfa_t1_baseline.toml`
- `uv run --locked scripts/run_experiment.py --config configs/exp/vfa_t1_rician.toml`
- `uv run --locked scripts/run_experiment.py --config configs/exp/vfa_t1_b1range_rician_outlier.toml`

比較レポート（集計CSV+比較図）：

- `uv run --locked scripts/compare_runs.py --runs output/runs/<run_id1> output/runs/<run_id2> output/runs/<run_id3>`

`run_experiment.py` は `metrics/*_per_sample.csv` も出力するため、比較レポートでは残差分布や `T1 true` / `B1` に対する誤差の層別図も生成されます。

さらに、`T1 true` / `B1` をビン分けした **層別 |誤差| 集計（mean / median / p95）** の図とCSV（`failure__abs_t1_err_by_*_bin.*` / `failure__abs_t1_err_p95_by_*_bin.png`）も出力します。

固定ビン（run間で同じ境界で比較したい場合）：

- `uv run --locked scripts/compare_runs.py --runs ... --t1-range 0.2 2.0 --t1-bins 10 --b1-range 0.8 1.2 --b1-bins 10`

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

## コミットメッセージ規約

- `<emoji> <type>(<scope>): <description>`
- 例：`✨ feat(core): add mono_t2 forward model`
