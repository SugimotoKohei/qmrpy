# 実データ I/O

qmrpy のコア機能は TIFF I/O のみで動作します。NIfTI、DICOM、BIDS 形式を扱う場合は optional dependency を追加してください。

```bash
pip install "qmrpy[io]"
# 開発環境では
uv sync --extra io
```

## NIfTI

`load_nifti()` は画像配列、affine、header を返します。フィット後のパラメータマップは `save_nifti_map()` に同じ affine/header を渡すことで、入力画像の空間メタデータを継承できます。

```python
from qmrpy.io import load_nifti, save_nifti_map
from qmrpy.models import T2Mono

data, affine, header = load_nifti("sub-01_echoes.nii.gz")
model = T2Mono(te_ms=[10, 20, 40, 80])
maps = model.fit_image(data, mask="otsu", n_jobs=-1)
save_nifti_map("sub-01_t2map.nii.gz", maps, "t2_ms", affine=affine, header=header)
```

## DICOM

`load_dicom_series()` はディレクトリ配下の DICOM ファイルを再帰的に読み込みます。単一コントラストは `(z, y, x)`、複数エコーや複数コントラストは `(z, y, x, n_volumes)` として返します。

```python
from qmrpy.io import load_dicom_series

data, meta = load_dicom_series("dicom/sub-01_t2")
print(data.shape)
print(meta["echo_time_ms"])
```

返却メタデータには、利用可能な範囲で `echo_time_ms`、`repetition_time_ms`、`flip_angle_deg`、`inversion_time_ms`、`voxel_spacing_mm`、簡易 affine が含まれます。

## BIDS

`load_bids_relaxometry()` は qMRI-BIDS で一般的な NIfTI + JSON sidecar の最小読み込みヘルパです。BIDS JSON の `EchoTime`、`RepetitionTime`、`InversionTime` は秒単位として読み、qmrpy 側では `_ms` キーへ変換します。

```python
from qmrpy.io import load_bids_relaxometry

data, meta = load_bids_relaxometry("bids/sub-01/anat", suffix="T2w")
print(meta["echo_time_ms"])
```

このヘルパは緩和時間解析に必要な 3D/4D NIfTI と主要取得パラメータの取得に限定しています。高度な BIDS validation や dataset 全体の索引化は対象外です。
