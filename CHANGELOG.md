# Changelog

このファイルは Keep a Changelog 形式に従います。

## [Unreleased]

## [2.0.0] - 2026-08-01

### Removed

- **BREAKING**: `qmrpy.models.MPPCA`（Marchenko-Pastur PCA denoising）とその
  テスト・検証スクリプト・config を削除した。実装が qMRLab の
  `External/mppca_denoise/MPdenoising.m` の翻訳であり、同ファイルは非商用主体に
  限定して use/copy/modify のみを許諾している（distribute / sublicense の許諾なし、
  臨床利用不可）。MIT ライセンスでの再頒布が許諾範囲を超えるため、再ライセンスでは
  なく削除を選択した。詳細は `THIRD_PARTY_NOTICES.md` の "Removed components" を参照。

### Changed

- **BREAKING**: `mrzerocore` を必須依存から optional extra `qmrpy[mrzero]` へ移した。
  MRzeroCore は AGPL-3.0 であり、MIT 配布との結合に copyleft が及ぶため。
  `qmrpy.sim.mrzero` は従来どおり遅延 import で動作し、extra 未導入時は導入案内を出す。
  副次的に torch を含む重い依存がデフォルトインストールから外れた。
- `THIRD_PARTY_NOTICES.md` を全面改訂し、qMRLab / DECAES.jl の移植元ファイルを
  モジュール単位で対応付け、依存パッケージのライセンス一覧と AGPL extra の注意を追記した。
- qMRLab / DECAES.jl から翻訳した各モジュールの先頭に帰属表示コメントを追加した。

## [1.1.0] - 2026-07-06

### Added

- NIfTI、DICOM、最小 qMRI-BIDS I/O helper を追加。
- `qmrpy` CLI を追加し、`info`、`fit`、`validate` を提供。
- T1rho spin-lock mapping を追加。
- MTR / MTsat mapping を追加。
- 辞書ベース MRF 同時 T1-T2 mapping を追加。
- 2プール近似の T2 water/fat separation を追加。
- ガバナンス文書（CONTRIBUTING、CHANGELOG、CODE_OF_CONDUCT、SECURITY、CITATION）を追加。
- pre-commit、mypy、coverage artifact、OS matrix を含む CI 品質ゲートを追加。

### Changed

- package keywords から未実装の diffusion を外し、relaxometry / magnetization transfer の実態に合わせた。
- core validation suite に T1rho、MT、MRF、T2 water/fat の synthetic recovery cases を追加。
- docs deploy workflow を locked dependency build に変更。

## [1.0.0] - 2026-01-15

### Added

- T1/T2/B0/B1/QSM/noise/simulation の主要モデルを統一 API で提供。
- `FitResult` による params 辞書互換 + `quality` / `diagnostics` メタデータスキーマを追加。
- core validation suite と JOSS 向け検証出力を整備。

[Unreleased]: https://github.com/SugimotoKohei/qmrpy/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/SugimotoKohei/qmrpy/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/SugimotoKohei/qmrpy/releases/tag/v1.0.0
